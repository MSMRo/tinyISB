# tinyISB Python Tooling & Machine Learning Pipeline

This directory contains the Python workflows, datasets, and training scripts managed with [`uv`](https://github.com/astral-sh/uv).

## Environment Setup with `uv`

To initialize the environment and install dependencies:

```bash
# Create virtual environment with Python 3.12 (or 3.11)
uv venv .venv --python 3.12

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install required packages
uv pip install --system-certs -r python/requirements.txt
```

---

## Machine Learning & Signal Processing Scripts

### 1. `01_train_dnn_basic.py`
- **Architecture**: Multi-Layer Perceptron (DNN: 4 inputs -> 16 -> 8 -> 3 outputs).
- **Task**: Multi-class sensory feature classification.
- **Quantization**: Full INT8 Post-Training Quantization with representative dataset.
- **Export Target**: `examples/BasicClassification/model_data.h`.

```bash
python python/01_train_dnn_basic.py
```

---

### 2. `02_train_ecg_1dcnn.py`
- **Data Generator**: `neurokit2.ecg_simulate()` generates realistic ECG time-series at 250 Hz.
- **Classes**:
  - Class 0: Normal Sinus Rhythm (NSR, HR ~72 bpm)
  - Class 1: Tachycardia / Ventricular Arrhythmia (HR ~135 bpm)
  - Class 2: Bradycardia / Ectopic beats (HR ~45 bpm)
- **Architecture**: 1D-CNN (`Conv1D(8, k=5, s=2)` -> `MaxPool1D(2)` -> `Conv1D(16, k=3, s=2)` -> `GAP` -> `Dense(12)` -> `Dense(3)`).
- **Quantization**: Full INT8 with ECG representative calibration.
- **Export Target**: `examples/ECGArrhythmiaClassifier/ecg_model_data.h`.

```bash
python python/02_train_ecg_1dcnn.py
```

---

### 3. `03_train_emg_knowledge_distillation.py`
- **Data Generator**: `neurokit2.emg_simulate()` generates multi-channel sEMG gestures at 1000 Hz:
  - Gesture 0: Rest / Baseline
  - Gesture 1: Fist / Hand Closure
  - Gesture 2: Wrist Flexion
  - Gesture 3: Wrist Extension
- **Teacher Model**: Deep hybrid `Conv1D + Bidirectional LSTM + Dense` (18k+ parameters).
- **Student Model**: Ultra-compact feed-forward DNN (`Dense(24) -> Dense(12) -> Dense(4)`, ~1,900 parameters).
- **Algorithm**: Hinton Knowledge Distillation with temperature $T=3.5$ and loss weighting $\alpha=0.7$.
- **Quantization**: Full INT8 for deployment on ultra-constrained MCUs.
- **Export Target**: `examples/EMGGestureClassifier/emg_model_data.h`.

```bash
python python/03_train_emg_knowledge_distillation.py
```

---

### 4. `04_train_signal_regressor.py`
- **Task**: Continuous bio-signal regression (predicting continuous respiratory rate / force output from multi-channel sensors).
- **Architecture**: DNN Regressor (`Dense(20) -> Dense(12) -> Dense(1)`).
- **Quantization**: Full INT8 Quantization.
- **Export Target**: `examples/SignalRegressor/regressor_model_data.h`.

```bash
python python/04_train_signal_regressor.py
```

---

## Model Exporter Tool (`tools/export_to_tinyisb.py`)

Convert any `.tflite`, `.keras`, or `.h5` model to a 16-byte aligned C++ header:

```bash
python tools/export_to_tinyisb.py --input my_model.tflite --output model_data.h --array-name model_data
```

Inspect metadata:
```bash
python tools/export_to_tinyisb.py --input my_model.tflite --output model_data.h --info
```
