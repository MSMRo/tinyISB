#!/usr/bin/env python3
"""
01_train_dnn_basic.py
=====================
Trains a lightweight Deep Neural Network (DNN / MLP) classifier for multi-class
sensory classification, applies full INT8 post-training quantization with a
representative dataset, and exports the model as a C++ header for tinyISB.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import exporter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.export_to_tinyisb import quantize_and_export_keras, inspect_tflite_bytes, print_model_info


def main():
    print("[1/5] Generating synthetic sensory feature dataset (4 features, 3 classes)...")
    np.random.seed(42)
    tf.random.set_seed(42)

    n_samples = 1500
    n_features = 4
    n_classes = 3

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(f"      Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"      Feature means: {scaler.mean_}")
    print(f"      Feature scales: {scaler.scale_}")

    print("\n[2/5] Constructing TinyML-optimized DNN architecture...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,), name="sensor_features"),
        tf.keras.layers.Dense(16, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(8, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(n_classes, activation="softmax", name="output_classes")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    print("\n[3/5] Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=35,
        batch_size=32,
        verbose=0
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"      Training finished -> Test Accuracy: {test_acc * 100:.2f}%")

    print("\n[4/5] Applying full INT8 post-training quantization with representative dataset...")
    def representative_dataset_gen():
        for i in range(100):
            yield [X_train[i:i+1]]

    output_header = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "examples", "BasicClassification", "model_data.h"
    ))

    tflite_bytes = quantize_and_export_keras(
        model,
        representative_data_gen=representative_dataset_gen,
        quantize_int8=True,
        output_header_path=output_header,
        array_name="model_data"
    )

    print(f"\n[5/5] Validating exported TFLite INT8 model...")
    info = inspect_tflite_bytes(tflite_bytes)
    print_model_info(info)

    # Verify inference using TFLite interpreter
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
        # INT8 input scaling
        if inp_det["dtype"] == np.int8:
            q_input = np.round(sample / in_scale + in_zero).clip(-128, 127).astype(np.int8)
            interpreter.set_tensor(inp_det["index"], q_input)
        else:
            interpreter.set_tensor(inp_det["index"], sample)

        interpreter.invoke()
        q_out = interpreter.get_tensor(out_det["index"])[0]

        if out_det["dtype"] == np.int8:
            dequant_out = (q_out.astype(np.float32) - out_zero) * out_scale
        else:
            dequant_out = q_out

        pred_class = np.argmax(dequant_out)
        if pred_class == y_test[i]:
            correct += 1

    quant_acc = correct / len(X_test)
    print(f"      Quantized INT8 Model Accuracy: {quant_acc * 100:.2f}%")
    print(f"      Header written to: {output_header}")

    # Also print sample test input for Arduino verification
    sample_feat = X_test[0]
    sample_label = y_test[0]
    print("\n--- Test Sample for Arduino Verification ---")
    print(f"Input Features: {{{', '.join(f'{f:.4f}f' for f in sample_feat)}}}")
    print(f"True Class: {sample_label}")
    print(f"Predicted Class: {np.argmax(model.predict(X_test[0:1], verbose=0))}")


if __name__ == "__main__":
    main()
