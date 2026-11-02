/**
 * @file EMGGestureClassifier.ino
 * @brief Distilled Surface EMG (sEMG) Hand Gesture Classifier using tinyISB
 *
 * Model:
 *  - Distilled Student Feed-Forward DNN (Knowledge Distilled from BiLSTM Teacher)
 *  - Input: 64-sample EMG window (fs = 1000 Hz, 64 ms window)
 *  - Output: 4 Muscular Gestures:
 *      0: Rest / Neutral Tone
 *      1: Fist / Hand Closure
 *      2: Wrist Flexion
 *      3: Wrist Extension
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
#include "emg_model_data.h"

// 64 raw sEMG samples
constexpr size_t kEMGWindowSize = 64;
constexpr size_t kTensorArenaSize = 6 * 1024; // 6 KB arena
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// tinyISB High-Level Classifier
static tinyisb::Classifier emg_classifier;

// Gesture semantic labels
static const char* kGestureLabels[4] = {
    "Rest / Neutral",
    "Fist / Hand Closure",
    "Wrist Flexion",
    "Wrist Extension"
};

// Real synthetic sEMG feature vector for test verification
static const float kSampleEMGFist[kEMGWindowSize] = {
     0.12f, -0.45f,  0.89f, -1.20f,  1.65f, -1.90f,  2.10f, -1.85f,
     1.45f, -0.95f,  0.60f, -0.30f,  0.15f, -0.05f,  0.40f, -0.75f,
     1.10f, -1.55f,  1.95f, -2.25f,  2.40f, -2.15f,  1.70f, -1.10f,
     0.70f, -0.35f,  0.20f, -0.10f,  0.05f,  0.30f, -0.65f,  1.05f,
    -1.45f,  1.80f, -2.05f,  2.15f, -1.95f,  1.50f, -0.90f,  0.55f,
    -0.25f,  0.10f, -0.05f,  0.25f, -0.55f,  0.90f, -1.30f,  1.65f,
    -1.85f,  1.90f, -1.65f,  1.25f, -0.75f,  0.40f, -0.20f,  0.08f,
    -0.02f,  0.15f, -0.40f,  0.75f, -1.10f,  1.35f, -1.40f,  1.15f
};

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}

    Serial.println();
    Serial.println(F("=============================================================="));
    Serial.println(F("   tinyISB: Distilled sEMG Hand Gesture Classifier (TinyML)   "));
    Serial.println(F("=============================================================="));

    // Initialize distilled student model
    Serial.println(F("[tinyISB] Initializing Distilled EMG Student Model..."));
    tinyisb::Status status = emg_classifier.begin(emg_model_data, tensor_arena, kTensorArenaSize);

    if (status != tinyisb::Status::OK) {
        Serial.print(F("[FATAL] EMG Classifier initialization failed: "));
        Serial.println(tinyisb::status_to_string(status));
        while (true) { delay(1000); }
    }

    Serial.println(F("[tinyISB] EMG Model initialized successfully!"));
    emg_classifier.print_metadata();

    Serial.println(F("\n--- Evaluating EMG Hand Gesture Test Window ---"));
    uint32_t t_start = micros();
    int gesture_idx = emg_classifier.predict(kSampleEMGFist, kEMGWindowSize);
    uint32_t t_inference = micros() - t_start;

    Serial.print(F("Recognized Gesture : [Class ")); Serial.print(gesture_idx); Serial.print(F("] "));
    if (gesture_idx >= 0 && gesture_idx < 4) {
        Serial.println(kGestureLabels[gesture_idx]);
    } else {
        Serial.println(F("Unknown"));
    }
    Serial.print(F("Inference Latency  : ")); Serial.print(t_inference); Serial.println(F(" us"));

    Serial.println(F("\nGesture Confidence Distribution (predict_proba):"));
    for (size_t g = 0; g < 4; ++g) {
        float prob = emg_classifier.predict_proba(kSampleEMGFist, g);
        Serial.print(F("  - ")); Serial.print(kGestureLabels[g]);
        Serial.print(F(": ")); Serial.print(prob * 100.0f, 2); Serial.println(F("%"));
    }
    Serial.println(F("==============================================================\n"));
}

void loop() {
    static uint32_t step = 0;
    step++;

    float emg_window[kEMGWindowSize];
    float scale_factor = 0.8f + (float)(step % 5) * 0.1f;

    for (size_t i = 0; i < kEMGWindowSize; ++i) {
        emg_window[i] = kSampleEMGFist[i] * scale_factor;
    }

    uint32_t t0 = micros();
    int g_pred = emg_classifier.predict(emg_window, kEMGWindowSize);
    uint32_t dt = micros() - t0;
    float conf = emg_classifier.predict_proba(emg_window, g_pred);

    Serial.print(F("[Frame #")); Serial.print(step);
    Serial.print(F("] -> "));
    if (g_pred >= 0 && g_pred < 4) {
        Serial.print(kGestureLabels[g_pred]);
    }
    Serial.print(F(" | Confidence: ")); Serial.print(conf * 100.0f, 1);
    Serial.print(F("% | Time: ")); Serial.print(dt); Serial.println(F(" us"));

    delay(1500);
}
