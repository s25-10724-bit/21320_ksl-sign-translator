import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
import mediapipe as mp

from config import (
    DEFAULT_MAX_FRAMES,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
    STABLE_COUNT,
)

from extract_landmarks import extract_frame_landmarks, resample_or_pad


def load_labels(label_path):
    label_path = Path(label_path)

    labels = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return labels


def load_scaler(scaler_path):
    data = np.load(scaler_path)

    mean = data["mean"].astype(np.float32)
    scale = data["scale"].astype(np.float32)

    scale[scale == 0] = 1.0

    return mean, scale


def preprocess_sequence(sequence, max_frames, mean, scale):
    sequence = resample_or_pad(
        np.asarray(sequence, dtype=np.float32),
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


def speak(tts_engine, text):
    tts_engine.say(text)
    tts_engine.runAndWait()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="models/ksl_model.tflite")
    parser.add_argument("--labels", default="models/labels.txt")
    parser.add_argument("--scaler", default="models/scaler.npz")
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--camera", type=int, default=0)

    args = parser.parse_args()

    labels = load_labels(args.labels)
    mean, scale = load_scaler(args.scaler)
    classifier = TFLiteClassifier(args.model)

    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 165)

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    sequence = []

    last_label = None
    stable_counter = 0
    spoken_label = None
    last_speak_time = 0

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    print("프로그램 실행 중")
    print("종료하려면 q를 누르세요.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        while True:
            success, frame = cap.read()

            if not success:
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

            if len(sequence) > args.max_frames:
                sequence = sequence[-args.max_frames:]

            predicted_text = "인식 대기 중"
            confidence = 0.0

            if len(sequence) >= max(5, args.max_frames // 3):
                x = preprocess_sequence(
                    sequence,
                    args.max_frames,
                    mean,
                    scale
                )

                probabilities = classifier.predict(x)

                predicted_index = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_index])
                predicted_label = labels[predicted_index]

                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_text = predicted_label

                    if predicted_label == last_label:
                        stable_counter += 1
                    else:
                        stable_counter = 1
                        last_label = predicted_label

                    current_time = time.time()

                    if (
                        stable_counter >= STABLE_COUNT
                        and predicted_label != spoken_label
                        and current_time - last_speak_time > 2.0
                    ):
                        spoken_label = predicted_label
                        last_speak_time = current_time

                        speak(tts_engine, predicted_label)

                else:
                    predicted_text = "확신 부족"
                    stable_counter = 0
                    last_label = None

            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 80),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"{predicted_text} / confidence: {confidence:.2f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("KSL Sign Translator", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
