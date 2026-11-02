#!/usr/bin/env python3
"""
04_train_signal_regressor.py
============================
Trains a continuous signal regression Deep Neural Network (DNN) for TinyML.
Predicts continuous physiological parameters (e.g. continuous respiratory rate /
muscle force output) from multi-channel sensor features.

Quantizes to INT8 and exports to C++ header for tinyISB Regressor inference.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.export_to_tinyisb import quantize_and_export_keras, inspect_tflite_bytes, print_model_info


def generate_regression_data(n_samples=1500, n_features=6):
    """
    Generate synthetic multi-feature physiological continuous response data.
    E.g. estimating exertion force or respiration rate from bio-impedance & accelerometer features.
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Non-linear relationship with noise
    y = (
        2.5 * X[:, 0]
        - 1.8 * np.sin(X[:, 1] * 2.0)
        + 0.9 * (X[:, 2] ** 2)
        + 1.2 * X[:, 3] * X[:, 4]
        + np.random.normal(0, 0.15, size=n_samples)
    ).astype(np.float32)

    # Scale y to realistic range (e.g., 10.0 to 35.0 breaths per minute)
    y = ((y - y.min()) / (y.max() - y.min())) * 25.0 + 10.0
    return X, y


def main():
    print("[1/5] Synthesizing continuous bio-signal regression dataset...")
    n_features = 6
    X, y = generate_regression_data(n_samples=1600, n_features=n_features)
    print(f"      Dataset: X shape={X.shape}, y shape={y.shape}")
    print(f"      Target range: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[2/5] Constructing Regression DNN Architecture...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,), name="sensor_inputs"),
        tf.keras.layers.Dense(20, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(12, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(1, activation="linear", name="continuous_output")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()

    print("\n[3/5] Training Regression Model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=32,
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"      Float32 Model Test MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    print("\n[4/5] Applying full INT8 Quantization with representative dataset...")
    def rep_gen():
        for i in range(100):
            yield [X_train[i:i+1]]

    output_header = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "examples", "SignalRegressor", "regressor_model_data.h"
    ))

    tflite_bytes = quantize_and_export_keras(
        model,
        representative_data_gen=rep_gen,
        quantize_int8=True,
        output_header_path=output_header,
        array_name="regressor_model_data"
    )

    print("\n[5/5] Inspecting and verifying Quantized INT8 Regressor...")
    info = inspect_tflite_bytes(tflite_bytes)
    print_model_info(info)

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_scale, in_zero = inp_det["quantization"]
    out_scale, out_zero = out_det["quantization"]

    q_preds = []
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
            val = (q_out.astype(np.float32) - out_zero) * out_scale
        else:
            val = q_out
        q_preds.append(float(val[0]))

    q_mse = mean_squared_error(y_test, q_preds)
    q_r2 = r2_score(y_test, q_preds)
    print(f"      INT8 Quantized Regressor MSE: {q_mse:.4f}, R2 Score: {q_r2:.4f}")
    print(f"      Model written to: {output_header}")

    # Print sample test vector for Arduino verification
    print("\nSample test vector for Arduino:")
    print(f"Features: {{{', '.join(f'{v:.4f}f' for v in X_test[0])}}}")
    print(f"Expected Target Value: {y_test[0]:.4f}")
    print(f"INT8 Prediction: {q_preds[0]:.4f}")


if __name__ == "__main__":
    main()
