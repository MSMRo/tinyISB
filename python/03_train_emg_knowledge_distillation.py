#!/usr/bin/env python3
"""
03_train_emg_knowledge_distillation.py
======================================
Knowledge Distillation in PyTorch + TensorFlow Lite:
Trains a heavy Recurrent/Convolutional Teacher model on synthetic sEMG signals
generated with NeuroKit2, and distills the learned representations into a
compact, ultra-fast feed-forward Student DNN.

Exports the distilled Student model to INT8 TFLite and C++ header for tinyISB.
"""

import os
import sys
import numpy as np
import neurokit2 as nk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.export_to_tinyisb import quantize_and_export_keras, inspect_tflite_bytes, print_model_info


def generate_emg_data(sampling_rate=1000, window_size=64, samples_per_class=300):
    """
    Generate synthetic sEMG gesture signals using NeuroKit2.
    """
    print(f"Generating synthetic sEMG signals (fs={sampling_rate} Hz, window={window_size})...")
    emg_rest = nk.emg_simulate(duration=60, sampling_rate=sampling_rate, burst_number=0, noise=0.005, random_state=42)
    emg_fist = nk.emg_simulate(duration=60, sampling_rate=sampling_rate, burst_number=25, burst_duration=1.2, noise=0.05, random_state=43)
    emg_flex = nk.emg_simulate(duration=60, sampling_rate=sampling_rate, burst_number=18, burst_duration=0.8, noise=0.03, random_state=44)
    emg_ext  = nk.emg_simulate(duration=60, sampling_rate=sampling_rate, burst_number=12, burst_duration=2.0, noise=0.04, random_state=45)

    X, y = [], []
    def extract_windows(signal, label):
        step = window_size // 2
        count = 0
        arr = np.asarray(signal, dtype=np.float32)
        for start in range(0, len(arr) - window_size, step):
            if count >= samples_per_class:
                break
            seg = arr[start : start + window_size]
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-6)
            X.append(seg)
            y.append(label)
            count += 1

    extract_windows(emg_rest, 0)
    extract_windows(emg_fist, 1)
    extract_windows(emg_flex, 2)
    extract_windows(emg_ext, 3)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# --- PyTorch Models ---

class TeacherModel(nn.Module):
    """
    Deep Conv1D + Bidirectional LSTM Teacher Model
    """
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (batch, 64) -> (batch, 1, 64)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.conv(x)  # (batch, 32, 32)
        feat = feat.permute(0, 2, 1)  # (batch, 32, 32)
        out, _ = self.lstm(feat)  # (batch, 32, 64)
        last_out = out[:, -1, :]  # (batch, 64)
        return self.fc(last_out)


class StudentDNN(nn.Module):
    """
    Lightweight feed-forward DNN for microcontroller deployment
    """
    def __init__(self, in_features=64, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_b, y_b in loader:
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
        correct += (out.argmax(dim=1) == y_b).sum().item()
        total += len(y_b)
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_b, y_b in loader:
            out = model(x_b)
            correct += (out.argmax(dim=1) == y_b).sum().item()
            total += len(y_b)
    return correct / total


def main():
    print("[1/5] Synthesizing 4-gesture sEMG dataset with NeuroKit2...")
    window_size = 64
    n_classes = 4
    X, y = generate_emg_data(sampling_rate=1000, window_size=window_size, samples_per_class=350)
    print(f"      Total samples: {len(X)}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)), batch_size=32, shuffle=False)

    print("\n[2/5] Training Heavy Teacher Model (Conv1D + BiLSTM)...")
    teacher = TeacherModel(in_channels=1, num_classes=n_classes)
    t_optimizer = torch.optim.Adam(teacher.parameters(), lr=0.003)
    criterion_ce = nn.CrossEntropyLoss()

    for epoch in range(1, 25):
        train_epoch(teacher, train_loader, t_optimizer, criterion_ce)

    teacher_acc = evaluate(teacher, test_loader)
    print(f"      Teacher Model Test Accuracy: {teacher_acc * 100:.2f}%")

    print("\n[3/5] Distilling Teacher Knowledge into Lightweight Student DNN...")
    student = StudentDNN(in_features=window_size, num_classes=n_classes)
    s_optimizer = torch.optim.Adam(student.parameters(), lr=0.005)

    T = 3.5      # Temperature
    alpha = 0.7  # Distillation loss weight
    kl_div = nn.KLDivLoss(reduction="batchmean")

    teacher.eval()
    for epoch in range(1, 35):
        student.train()
        for x_b, y_b in train_loader:
            s_optimizer.zero_grad()
            with torch.no_grad():
                t_logits = teacher(x_b)

            s_logits = student(x_b)
            loss_ce = criterion_ce(s_logits, y_b)
            loss_kd = kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1)
            ) * (T * T)

            loss = alpha * loss_kd + (1.0 - alpha) * loss_ce
            loss.backward()
            s_optimizer.step()

    student_acc = evaluate(student, test_loader)
    print(f"      Distilled Student DNN Test Accuracy: {student_acc * 100:.2f}%")

    print("\n[4/5] Porting Student DNN to Keras for TFLite Micro Conversion...")
    # Mirror Student architecture in Keras
    keras_student = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size,), name="emg_input"),
        tf.keras.layers.Dense(24, activation="relu", name="fc1"),
        tf.keras.layers.Dense(12, activation="relu", name="fc2"),
        tf.keras.layers.Dense(n_classes, activation="softmax", name="gesture_class")
    ])

    # Copy distilled weights from PyTorch to Keras
    w1 = student.fc1.weight.detach().cpu().numpy().T
    b1 = student.fc1.bias.detach().cpu().numpy()
    w2 = student.fc2.weight.detach().cpu().numpy().T
    b2 = student.fc2.bias.detach().cpu().numpy()
    w3 = student.fc3.weight.detach().cpu().numpy().T
    b3 = student.fc3.bias.detach().cpu().numpy()

    keras_student.layers[0].set_weights([w1, b1])
    keras_student.layers[1].set_weights([w2, b2])
    keras_student.layers[2].set_weights([w3, b3])

    print("\n[5/5] Full INT8 Quantization and C++ Header Export...")
    def rep_gen():
        for i in range(100):
            yield [X_tr[i:i+1].astype(np.float32)]

    output_header = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "examples", "EMGGestureClassifier", "emg_model_data.h"
    ))

    tflite_bytes = quantize_and_export_keras(
        keras_student,
        representative_data_gen=rep_gen,
        quantize_int8=True,
        output_header_path=output_header,
        array_name="emg_model_data"
    )

    info = inspect_tflite_bytes(tflite_bytes)
    print_model_info(info)

    # Verify quantized inference
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_scale, in_zero = inp_det["quantization"]
    out_scale, out_zero = out_det["quantization"]
    correct = 0

    for i in range(len(X_te)):
        sample = X_te[i:i+1].astype(np.float32)
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

        if np.argmax(deq) == y_te[i]:
            correct += 1

    print(f"      INT8 Quantized Student Accuracy: {(correct / len(X_te)) * 100:.2f}%")
    print(f"      Model written to: {output_header}")


if __name__ == "__main__":
    main()
