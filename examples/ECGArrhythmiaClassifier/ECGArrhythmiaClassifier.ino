/**
 * @file ECGArrhythmiaClassifier.ino
 * @brief 1D-CNN Real-Time ECG Arrhythmia Classifier using tinyISB
 *
 * Model:
 *  - 1D Convolutional Neural Network (Conv1D -> MaxPool -> Conv1D -> GAP -> Dense)
 *  - Input: 128-sample ECG window (fs = 250 Hz, 512 ms window)
 *  - Output: 3 Cardiac Conditions:
 *      0: Normal Sinus Rhythm (NSR)
 *      1: Tachycardia / Ventricular Arrhythmia
 *      2: Bradycardia / Irregular Heartbeat
 *  - Full INT8 Quantization (weights and activations)
 *
 * Cross-Platform Target:
 *  - Raspberry Pi Pico (RP2040) & Pico 2 (RP2350 Cortex-M33)
 *  - Espressif ESP32-S3 (Xtensa LX7) & ESP32-C6 (RISC-V)
 *  - Arduino Nano 33 BLE Sense (nRF52840 Cortex-M4F)
 */

#include <Arduino.h>

#if defined(ARDUINO_ARDUINO_NANO33BLE)
  #include <TensorFlowLite.h>
#else
  #include <tensorflow-runtime-universal.h>
#endif

#include <tinyISB.h>
#include "ecg_model_data.h"

// 128 samples at 250 Hz = ~512 ms cardiac window
constexpr size_t kECGWindowSize = 128;
constexpr size_t kTensorArenaSize = 10 * 1024; // 10 KB arena for 1D-CNN activations
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// tinyISB High-Level Classifier
static tinyisb::Classifier ecg_classifier;

// Condition labels
static const char* kCardiacLabels[3] = {
    "Normal Sinus Rhythm (NSR)",
    "Tachycardia / Ventricular Arrhythmia",
    "Bradycardia / Irregular Heartbeat"
};

// Real synthetic ECG segment test vector (Normal Sinus Rhythm)
static const float kNormalECGSample[kECGWindowSize] = {
     3.2463f,  1.9740f,  0.7536f, -0.3008f, -1.1305f, -1.7035f, -2.0176f, -2.1004f,
    -1.9961f, -1.7483f, -1.4013f, -0.9996f, -0.5841f, -0.1916f,  0.1471f,  0.4109f,
     0.5888f,  0.6770f,  0.6778f,  0.6000f,  0.4578f,  0.2676f,  0.0483f, -0.1802f,
    -0.3957f, -0.5772f, -0.7093f, -0.7813f, -0.7884f, -0.7303f, -0.6133f, -0.4485f,
    -0.2505f, -0.0354f,  0.1772f,  0.3662f,  0.5134f,  0.6052f,  0.6318f,  0.5902f,
     0.4852f,  0.3276f,  0.1325f, -0.0818f, -0.2882f, -0.4674f, -0.6044f, -0.6876f,
    -0.7092f, -0.6657f, -0.5606f, -0.4035f, -0.2084f,  0.0084f,  0.2198f,  0.4036f,
     0.5404f,  0.6150f,  0.6179f,  0.5463f,  0.4070f,  0.2131f, -0.0170f, -0.2599f,
    -0.4891f, -0.6791f, -0.8066f, -0.8551f, -0.8176f, -0.6972f, -0.5057f, -0.2619f,
     0.0076f,  0.2709f,  0.4996f,  0.6687f,  0.7584f,  0.7580f,  0.6656f,  0.4883f,
     0.2407f, -0.0514f, -0.3547f, -0.6382f, -0.8711f, -1.0264f, -1.0849f, -1.0360f,
    -0.8803f, -0.6276f, -0.3005f,  0.0673f,  0.4357f,  0.7604f,  0.9996f,  1.1189f,
     1.0965f,  0.9298f,  0.6312f,  0.2269f, -0.2443f, -0.7301f, -1.1738f, -1.5173f,
    -1.7161f, -1.7397f, -1.5794f, -1.2483f, -0.7788f, -0.2185f,  0.3725f,  0.9329f,
     1.3965f,  1.7032f,  1.8080f,  1.6888f,  1.3533f,  0.8402f,  0.2163f, -0.4343f,
    -1.0253f, -1.4889f, -1.7709f, -1.8340f, -1.6669f, -1.2858f, -0.7337f, -0.0763f
};

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}

    Serial.println();
    Serial.println(F("=============================================================="));
    Serial.println(F("    tinyISB: 1D-CNN Real-Time ECG Arrhythmia Classifier       "));
    Serial.println(F("=============================================================="));

    // Initialize 1D-CNN with external static arena
    Serial.println(F("[tinyISB] Initializing ECG 1D-CNN model..."));
    tinyisb::Status status = ecg_classifier.begin(ecg_model_data, tensor_arena, kTensorArenaSize);

    if (status != tinyisb::Status::OK) {
        Serial.print(F("[FATAL] ECG Classifier init error: "));
        Serial.println(tinyisb::status_to_string(status));
        while (true) { delay(1000); }
    }

    Serial.println(F("[tinyISB] ECG Model initialized successfully!"));
    ecg_classifier.print_metadata();

    Serial.println(F("\n--- Evaluating ECG Test Window ---"));
    uint32_t t_start = micros();
    int pred_class = ecg_classifier.predict(kNormalECGSample, kECGWindowSize);
    uint32_t t_inference = micros() - t_start;

    Serial.print(F("Diagnosis Result : [Class ")); Serial.print(pred_class); Serial.print(F("] "));
    if (pred_class >= 0 && pred_class < 3) {
        Serial.println(kCardiacLabels[pred_class]);
    } else {
        Serial.println(F("Unknown"));
    }
    Serial.print(F("Inference Latency: ")); Serial.print(t_inference); Serial.println(F(" us"));

    Serial.println(F("\nCardiac Condition Probability Distribution:"));
    for (size_t c = 0; c < 3; ++c) {
        float prob = ecg_classifier.predict_proba(kNormalECGSample, c);
        Serial.print(F("  - ")); Serial.print(kCardiacLabels[c]);
        Serial.print(F(": ")); Serial.print(prob * 100.0f, 2); Serial.println(F("%"));
    }
    Serial.println(F("==============================================================\n"));
}

void loop() {
    static uint32_t beat_count = 0;
    beat_count++;

    // Simulate real-time continuous buffering of ECG samples
    float window_buffer[kECGWindowSize];
    float phase_shift = (float)(beat_count % 16) * 0.05f;

    for (size_t i = 0; i < kECGWindowSize; ++i) {
        window_buffer[i] = kNormalECGSample[(i + (beat_count % 8)) % kECGWindowSize] + phase_shift;
    }

    uint32_t t0 = micros();
    int diagnosis = ecg_classifier.predict(window_buffer, kECGWindowSize);
    uint32_t dt = micros() - t0;
    float confidence = ecg_classifier.predict_proba(window_buffer, diagnosis);

    Serial.print(F("[Beat #")); Serial.print(beat_count);
    Serial.print(F("] -> "));
    if (diagnosis >= 0 && diagnosis < 3) {
        Serial.print(kCardiacLabels[diagnosis]);
    }
    Serial.print(F(" | Confidence: ")); Serial.print(confidence * 100.0f, 1);
    Serial.print(F("% | Exec: ")); Serial.print(dt); Serial.println(F(" us"));

    delay(1000);
}
