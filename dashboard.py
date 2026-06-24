import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf

from config import (
    DEFAULT_MAX_FRAMES,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
    STABLE_COUNT,
)

from extract_landmarks import extract_frame_landmarks, resample_or_pad
from label_map import to_display_label


MODEL_PATH = Path("models/ksl_model.tflite")
LABEL_PATH = Path("models/labels.txt")
SCALER_PATH = Path("models/scaler.npz")


def load_labels(path):
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            label = line.strip()

            if label:
                labels.append(label)

    return labels


def load_scaler(path):
    data = np.load(path)

    mean = data["mean"].astype(np.float32)
    scale = data["scale"].astype(np.float32)

    scale[scale == 0] = 1.0

    return mean, scale


def preprocess_sequence(sequence, max_frames, mean, scale):
    sequence = np.asarray(sequence, dtype=np.float32)

    sequence = resample_or_pad(
        sequence,
        max_frames=max_frames
    )

    x = sequence.reshape(1, -1)

    x = (x - mean) / scale

    return x.astype(np.float32)


class TFLiteClassifier:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, x):
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            x
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        return output[0]


def check_model_files():
    missing_files = []

    if not MODEL_PATH.exists():
        missing_files.append(str(MODEL_PATH))

    if not LABEL_PATH.exists():
        missing_files.append(str(LABEL_PATH))

    if not SCALER_PATH.exists():
        missing_files.append(str(SCALER_PATH))

    return missing_files


def main():
    st.set_page_config(
        page_title="KSL 수어 번역 대시보드",
        page_icon="🖐️",
        layout="wide"
    )

    st.title("KSL 수어 번역 프로그램")
    st.write("웹캠으로 입력된 수어 동작을 인식하여 한국어 텍스트로 출력하는 대시보드입니다.")

    missing_files = check_model_files()

    if missing_files:
        st.error("모델 파일이 아직 없습니다.")
        st.write("먼저 Colab에서 학습을 끝낸 뒤 models 폴더를 프로젝트에 넣어야 합니다.")
        st.write("없는 파일 목록:")
        for file in missing_files:
            st.write("-", file)
        return

    labels = load_labels(LABEL_PATH)
    mean, scale = load_scaler(SCALER_PATH)
    classifier = TFLiteClassifier(MODEL_PATH)

    st.success("모델 파일을 정상적으로 불러왔습니다.")

    st.subheader("인식 가능한 단어")

    display_labels = [to_display_label(label) for label in labels]
    st.write(", ".join(display_labels))

    start_button = st.button("실시간 인식 시작")

    if not start_button:
        st.info("버튼을 누르면 웹캠 인식이 시작됩니다.")
        st.warning("실행을 멈추려면 Colab이 아니라 로컬 PC 터미널에서 Ctrl + C를 누르세요.")
        return

    video_area = st.empty()
    result_area = st.empty()
    confidence_area = st.empty()
    history_area = st.empty()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다. 이 대시보드는 Colab이 아니라 웹캠이 연결된 컴퓨터에서 실행해야 합니다.")
        return

    sequence = []

    last_label = None
    stable_counter = 0
    confirmed_label = "인식 대기 중"
    confirmed_confidence = 0.0

    history = []

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:

        while True:
            ok, frame = cap.read()

            if not ok:
                st.error("웹캠 프레임을 읽을 수 없습니다.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            draw_result = hands.process(rgb)

            if draw_result.multi_hand_landmarks:
                for hand_landmarks in draw_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            feature = extract_frame_landmarks(frame, hands)

            sequence.append(feature)

            if len(sequence) > DEFAULT_MAX_FRAMES:
                sequence = sequence[-DEFAULT_MAX_FRAMES:]

            current_label = "인식 대기 중"
            current_confidence = 0.0

            if len(sequence) >= max(5, DEFAULT_MAX_FRAMES // 3):
                x = preprocess_sequence(
                    sequence,
                    DEFAULT_MAX_FRAMES,
                    mean,
                    scale
                )

                probabilities = classifier.predict(x)

                predicted_index = int(np.argmax(probabilities))
                current_confidence = float(probabilities[predicted_index])
                raw_label = labels[predicted_index]
                current_label = to_display_label(raw_label)

                if current_confidence >= CONFIDENCE_THRESHOLD:
                    if raw_label == last_label:
                        stable_counter += 1
                    else:
                        stable_counter = 1
                        last_label = raw_label

                    if stable_counter >= STABLE_COUNT:
                        confirmed_label = current_label
                        confirmed_confidence = current_confidence

                        if not history or history[-1] != confirmed_label:
                            history.append(confirmed_label)

                        if len(history) > 10:
                            history = history[-10:]

                else:
                    stable_counter = 0
                    last_label = None
                    current_label = "확신 부족"

            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 80),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"{current_confidence:.2f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            video_area.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

            result_area.metric(
                label="확정 인식 결과",
                value=confirmed_label
            )

            confidence_area.progress(
                min(max(confirmed_confidence, 0.0), 1.0)
            )

            if history:
                history_area.write("최근 인식 기록:", " → ".join(history))
            else:
                history_area.write("최근 인식 기록 없음")

            time.sleep(0.03)

    cap.release()


if __name__ == "__main__":
    main()
