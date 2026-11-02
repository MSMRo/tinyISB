/**
 * @file EMGGestureClassifier.ino
 * @brief Distilled Surface EMG (sEMG) Hand Gesture Classifier using tinyISB
 *
 * ============================================================================
 * tinyISB: TinyML Scikit-Learn Inference Engine for Microcontrollers
 * ============================================================================
 *
 * Biomechanical Context & Knowledge Distillation Walkthrough:
 *  - Domain: Myoelectric hand gesture recognition for active prosthetics & robotics.
 *  - Signal Acquisition: Differential surface EMG (sEMG) sampled at 500 Hz.
 *  - Window: 64 raw samples (128 ms window), sufficient to detect muscular onset.
 *  - The TinyML Dilemma:
 *      * Recurrent BiLSTM models achieve high temporal accuracy on EMG bursts,
 *        but require persistent hidden state buffers (c_t, h_t), large Flash (>300 KB),
 *        and non-deterministic latency unacceptable on resource-constrained MCUs.
 *  - Knowledge Distillation Solution (Hinton et al.):
 *      * A bulky Server Teacher (Conv1D + 2-layer BiLSTM, ~80k params) transfers its
 *        dark knowledge soft probabilities (Temperature T=3.5, Alpha=0.7) to a
 *        compact Student MLP (64 -> 32 -> 16 -> 4, < 3k params).
 *      * The resulting model occupies only ~3.9 KB in Flash and executes in < 1 ms on MCU!
 *  - Target Gesture Classes:
 *      0: Rest / Neutral Tone (Baseline muscle tone, relaxation).
 *      1: Fist / Hand Closure (Co-contraction of flexor & extensor compartments).
 *      2: Wrist Flexion (Dominant contraction of flexor carpi radialis).
 *      3: Wrist Extension (Dominant contraction of extensor carpi radialis).
 *
 * Target Microcontrollers:
 *  - Raspberry Pi Pico (RP2040) & Pico 2 (RP2350 Cortex-M33)
 *  - Espressif ESP32-S3 (Xtensa LX7) & ESP32-C6 (RISC-V)
 *  - Arduino Nano 33 BLE Sense (nRF52840 Cortex-M4F)
 */

#include <Arduino.h>

// ============================================================================
// 1. Cross-Platform Runtime Provider Selection
// ============================================================================
#if defined(ARDUINO_ARDUINO_NANO33BLE)
  #include <TensorFlowLite.h>
#else
  #include <tensorflow-runtime-universal.h>
#endif

// Include tinyISB inference library
#include <tinyISB.h>

// Include the 16-byte aligned distilled student model C-array
#include "emg_model_data.h"

// ============================================================================
// 2. Buffer & Static Tensor Arena Definition
// ============================================================================
// 64 raw sEMG samples (~128 ms at 500 Hz)
constexpr size_t kEMGWindowSize = 64;

// Static activation buffer allocated in contiguous internal SRAM.
// The distilled feed-forward student MLP requires approximately 5-6 KB of arena memory.
// alignas(16) ensures 128-bit memory bus alignment for SIMD operations.
constexpr size_t kTensorArenaSize = 6 * 1024; // 6 KB static Tensor Arena
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// ============================================================================
// 3. Estimator & Semantic Labels
// ============================================================================
// High-level tinyISB Classifier estimator
static tinyisb::Classifier emg_classifier;

// Semantic labels for myoelectric prosthetic actuation
static const char* kGestureLabels[4] = {
    "Rest / Neutral",
    "Fist / Hand Closure",
    "Wrist Flexion",
    "Wrist Extension"
};

// Real synthetic sEMG feature vector for test verification (Simulating a Fist gesture).
// Features represent normalized motor unit action potential (MUAP) amplitudes.
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

    // ========================================================================
    // 4. Student Model Initialization & Schema Verification
    // ========================================================================
    // Validates FlatBuffers header, prepares op resolvers, and allocates
    // activation tensors inside the 6 KB static arena.
    Serial.println(F("[tinyISB] Initializing Distilled EMG Student Model..."));
    tinyisb::Status status = emg_classifier.begin(emg_model_data, tensor_arena, kTensorArenaSize);

    if (status != tinyisb::Status::OK) {
        Serial.print(F("[FATAL] EMG Classifier initialization failed: "));
        Serial.println(tinyisb::status_to_string(status));
        Serial.println(F("[FATAL] Ensure arena size is sufficient. Halting."));
        while (true) { delay(1000); }
    }

    Serial.println(F("[tinyISB] EMG Model initialized successfully!"));

    // Print tensor metadata: Input [1, 64], Output [1, 4]
    emg_classifier.print_metadata();

    // ========================================================================
    // 5. Benchmark Single Test Ingestion
    // ========================================================================
    Serial.println(F("\n--- Evaluating EMG Hand Gesture Test Window ---"));
    uint32_t t_start = micros();

    // Ingest 64 raw float samples into the distilled student model.
    // Automatically quantizes inputs to INT8, runs feed-forward inference, and returns argmax.
    int gesture_idx = emg_classifier.predict(kSampleEMGFist, kEMGWindowSize);
    uint32_t t_inference = micros() - t_start;

    Serial.print(F("Recognized Gesture : [Class ")); Serial.print(gesture_idx); Serial.print(F("] "));
    if (gesture_idx >= 0 && gesture_idx < 4) {
        Serial.println(kGestureLabels[gesture_idx]);
    } else {
        Serial.println(F("Unknown"));
    }
    Serial.print(F("Inference Latency  : ")); Serial.print(t_inference); Serial.println(F(" us"));

    // Extract softmax confidence scores via predict_proba()
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

    // ========================================================================
    // 6. Real-Time Streaming Ingestion & Prosthetic Control
    // ========================================================================
    // Modulate amplitude to simulate varying muscular contraction efforts
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

    // Prosthetic Actuation Logic:
    // Only actuate robotic servos if model confidence exceeds 75% threshold
    if (conf >= 0.75f) {
        switch (g_pred) {
            case 0: /* Neutral: Open prosthetic fingers gently */ break;
            case 1: /* Fist: Close all grip finger servos */ break;
            case 2: /* Flexion: Actuate wrist rotation servo counter-clockwise */ break;
            case 3: /* Extension: Actuate wrist rotation servo clockwise */ break;
            default: break;
        }
    }

    delay(1500);
}
