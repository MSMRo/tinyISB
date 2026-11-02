#!/usr/bin/env python3
"""
02_train_ecg_1dcnn.py
=====================
Generates realistic physiological ECG signal datasets using NeuroKit2 for 3
cardiac conditions (Normal Sinus Rhythm, Tachycardia, Bradycardia/Irregular).
Builds a 1D Convolutional Neural Network (1D-CNN), trains it, applies full INT8
quantization, and exports the model as a C++ header for tinyISB inference.
"""

import os
import sys
import numpy as np
import neurokit2 as nk
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.export_to_tinyisb import quantize_and_export_keras, inspect_tflite_bytes, print_model_info


def generate_ecg_dataset(sampling_rate=250, window_size=128, samples_per_class=400):
    """
    Generate synthetic ECG segments across 3 classes using neurokit2.
    Class 0: Normal Sinus Rhythm (NSR, HR ~ 72 bpm)
    Class 1: Tachycardia / Ventricular Arrhythmia (HR ~ 135 bpm, increased noise)
    Class 2: Bradycardia / Ectopic rhythm (HR ~ 45 bpm, irregular intervals)
    """
    print(f"Generating synthetic ECG signals with NeuroKit2 (fs={sampling_rate} Hz, window={window_size})...")

    # Class 0: Normal
    ecg_normal = nk.ecg_simulate(duration=240, sampling_rate=sampling_rate, heart_rate=72, random_state=42)
    # Class 1: Tachycardia
    ecg_tachy = nk.ecg_simulate(duration=240, sampling_rate=sampling_rate, heart_rate=135, noise=0.1, random_state=43)
    # Class 2: Bradycardia / Irregular
    ecg_brady = nk.ecg_simulate(duration=240, sampling_rate=sampling_rate, heart_rate=45, noise=0.08, random_state=44)

    X = []
    y = []

    def extract_windows(signal, label):
        step = window_size // 2
        count = 0
        for start in range(0, len(signal) - window_size, step):
            if count >= samples_per_class:
                break
            segment = np.asarray(signal[start : start + window_size], dtype=np.float32)
            # Z-score normalization per window
            seg_mean = np.mean(segment)
            seg_std = np.std(segment) + 1e-6
            segment = (segment - seg_mean) / seg_std
            X.append(segment)
            y.append(label)
            count += 1

    extract_windows(ecg_normal, 0)
    extract_windows(ecg_tachy, 1)
    extract_windows(ecg_brady, 2)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Reshape for 1D-CNN: (num_samples, window_size, 1)
    X = np.expand_dims(X, axis=-1)
    return X, y


def main():
    print("[1/5] Synthesizing multi-condition ECG signals with NeuroKit2...")
    window_size = 128
    X, y = generate_ecg_dataset(sampling_rate=250, window_size=window_size, samples_per_class=350)
    print(f"      Total dataset shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"      Train count: {len(X_train)}, Test count: {len(X_test)}")

    print("\n[2/5] Designing 1D-CNN TinyML architecture...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, 1), name="ecg_window"),
        tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=2, padding="same", activation="relu", name="conv1d_1"),
        tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1"),
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=2, padding="same", activation="relu", name="conv1d_2"),
        tf.keras.layers.GlobalAveragePooling1D(name="gap"),
        tf.keras.layers.Dense(12, activation="relu", name="dense_dense"),
        tf.keras.layers.Dense(3, activation="softmax", name="cardiac_condition")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    print("\n[3/5] Training 1D-CNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        verbose=0
    )
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"      1D-CNN Validation Accuracy: {acc * 100:.2f}%")

    print("\n[4/5] Full INT8 Quantization with representative ECG calibration set...")
    def rep_gen():
        for i in range(100):
            yield [X_train[i:i+1]]

    output_header = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "examples", "ECGArrhythmiaClassifier", "ecg_model_data.h"
    ))

    tflite_bytes = quantize_and_export_keras(
        model,
        representative_data_gen=rep_gen,
        quantize_int8=True,
        output_header_path=output_header,
        array_name="ecg_model_data"
    )

    print("\n[5/5] Inspecting and verifying quantized INT8 1D-CNN...")
    info = inspect_tflite_bytes(tflite_bytes)
    print_model_info(info)

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    correct = 0
    in_scale = inp_det["quantization"][0]
    in_zero = inp_det["quantization"][1]
    out_scale = out_det["quantization"][0]
    out_zero = out_det["quantization"][1]

    for i in range(len(X_test)):
        sample = X_test[i:i+1]
        if inp_det["dtype"] == np.int8:
            q_in = np.round(sample / in_scale + in_zero).clip(-128, 127).astype(np.int8)
            interpreter.set_tensor(inp_det["index"], q_in)
        else:
            interpreter.set_tensor(inp_det["index"], sample)

        interpreter.invoke()
        q_out = interpreter.get_tensor(out_det["index"])[0]

        if out_det["dtype"] == np.int8:
            deq = (q_out.astype(np.float32) - out_zero) * out_scale
        else:
            deq = q_out

        if np.argmax(deq) == y_test[i]:
            correct += 1

    print(f"      Quantized 1D-CNN INT8 Accuracy: {(correct / len(X_test)) * 100:.2f}%")
    print(f"      Model written to: {output_header}")

    # Generate sample array snippet for easy test in Arduino
    sample_flat = X_test[0].flatten()
    print("\nSample test window (first 8 values):")
    print(f"{{{', '.join(f'{v:.4f}f' for v in sample_flat[:8])}, ...}}")
    print(f"True Condition Label: {y_test[0]}")


if __name__ == "__main__":
    main()
