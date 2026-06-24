from pathlib import Path
import argparse
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

import tensorflow as tf

from extract_landmarks import extract_video_landmarks


VIDEO_EXTENSIONS = [".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV"]


def load_class_labels(class_label_path):
    with open(class_label_path, "rb") as f:
        class_labels = pickle.load(f)

    return class_labels


def find_video_files(folder_path):
    video_files = []

    for extension in VIDEO_EXTENSIONS:
        video_files.extend(folder_path.rglob(f"*{extension}"))

    return sorted(video_files)


def load_dataset(data_dir, class_label_path, max_frames):
    data_dir = Path(data_dir)
    class_label_path = Path(class_label_path)

    class_labels = load_class_labels(class_label_path)

    X = []
    y = []

    class_folders = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    print("찾은 클래스 폴더 수:", len(class_folders))

    for class_folder in class_folders:
        folder_name = class_folder.name

        try:
            class_id = int(folder_name)
        except ValueError:
            print("숫자 폴더가 아니어서 건너뜀:", class_folder)
            continue

        if class_id not in class_labels:
            print("class_label.p에 없는 번호라서 건너뜀:", class_id)
            continue

        label_name = str(class_labels[class_id]).strip()

        if label_name == "":
            print("빈 라벨이라서 건너뜀:", class_id)
            continue

        video_files = find_video_files(class_folder)

        print(f"[{folder_name}] {label_name} 영상 개수:", len(video_files))

        for video_path in video_files:
            try:
                features = extract_video_landmarks(video_path, max_frames=max_frames)
                X.append(features.reshape(-1))
                y.append(label_name)
            except Exception as e:
                print("영상 처리 실패:", video_path, e)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y


def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_scaler(scaler, output_path):
    np.savez(
        output_path,
        mean=scaler.mean_,
        scale=scaler.scale_
    )


def convert_to_tflite(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="downloaded_data/Video",
        help="01, 02, 03 폴더가 들어 있는 Video 폴더 경로"
    )

    parser.add_argument(
        "--class_label",
        default="downloaded_data/Label/class_label.p",
        help="class_label.p 파일 경로"
    )

    parser.add_argument(
        "--output_dir",
        default="models",
        help="학습된 모델 저장 폴더"
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=30,
        help="영상 하나에서 사용할 프레임 수"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="학습 반복 횟수"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("데이터 불러오는 중")
    X, y = load_dataset(
        data_dir=args.data_dir,
        class_label_path=args.class_label,
        max_frames=args.max_frames
    )

    print("전체 데이터 개수:", len(X))
    print("클래스 개수:", len(set(y)))

    if len(X) == 0:
        raise ValueError("학습할 영상 데이터가 없습니다. downloaded_data/Video 폴더를 확인하세요.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    with open(output_dir / "labels.txt", "w", encoding="utf-8") as f:
        for label in label_encoder.classes_:
            f.write(label + "\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    save_scaler(scaler, output_dir / "scaler.npz")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    model = build_model(
        input_dim=X_train.shape[1],
        num_classes=len(label_encoder.classes_)
    )

    print("모델 학습 시작")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=16
    )

    print("모델 평가")

    y_pred = model.predict(X_test).argmax(axis=1)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )
    )

    keras_path = output_dir / "ksl_model.keras"
    tflite_path = output_dir / "ksl_model.tflite"

    model.save(keras_path)
    convert_to_tflite(model, tflite_path)

    print("저장 완료")
    print("Keras 모델:", keras_path)
    print("TFLite 모델:", tflite_path)
    print("라벨 파일:", output_dir / "labels.txt")
    print("스케일러 파일:", output_dir / "scaler.npz")


if __name__ == "__main__":
    main()
