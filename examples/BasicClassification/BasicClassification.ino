/**
 * @file BasicClassification.ino
 * @brief Basic Multi-Class Classification Example using tinyISB
 *
 * Demonstrates:
 *  - External Tensor Arena buffer allocation (zero heap fragmentation).
 *  - Model initialization with automatic INT8 quantization handling.
 *  - High-level Scikit-Learn API: begin(), predict(), predict_proba().
 *  - Inference execution benchmark timing (latency in microseconds).
 *
 * Supported Boards:
 *  - Raspberry Pi Pico (RP2040) & Pico 2 (RP2350 Cortex-M33)
 *  - Espressif ESP32-S3 (Xtensa LX7) & ESP32-C6 (RISC-V)
 *  - Arduino Nano 33 BLE Sense (nRF52840 Cortex-M4F)
 */

#include <Arduino.h>

// Microcontroller TFLM runtime provider
#if defined(ARDUINO_ARDUINO_NANO33BLE)
  #include <TensorFlowLite.h>
#else
  #include <tensorflow-runtime-universal.h>
#endif

#include <tinyISB.h>
#include "model_data.h"

// Contiguous Tensor Arena buffer in static RAM (16-byte aligned)
constexpr size_t kTensorArenaSize = 4 * 1024; // 4 KB is plenty for this model
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// tinyISB High-Level Classifier estimator
static tinyisb::Classifier classifier;

// Sample sensory test feature vector from test dataset:
// True expected class: 2
static const float kSampleFeatures[4] = {0.9375f, -0.3574f, 0.3241f, 0.4415f};

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {
        // Wait for serial monitor connection if available
    }

    Serial.println();
    Serial.println(F("=================================================="));
    Serial.println(F("     tinyISB: Basic Classification Example        "));
    Serial.println(F("=================================================="));

    // Initialize model with external contiguous arena
    Serial.println(F("[tinyISB] Initializing model engine..."));
    tinyisb::Status status = classifier.begin(model_data, tensor_arena, kTensorArenaSize);

    if (status != tinyisb::Status::OK) {
        Serial.print(F("[ERROR] Classifier initialization failed: "));
        Serial.println(tinyisb::status_to_string(status));
        while (true) {
            delay(1000);
        }
    }

    Serial.println(F("[tinyISB] Model initialized successfully!"));
    // Print comprehensive metadata inspection
    classifier.print_metadata();

    Serial.println(F("\n--- Running Benchmark Single Test Vector ---"));
    Serial.print(F("Test Input: ["));
    for (size_t i = 0; i < 4; ++i) {
        Serial.print(kSampleFeatures[i], 4);
        if (i < 3) Serial.print(F(", "));
    }
    Serial.println(F("] (Expected: Class 2)"));

    // Measure inference latency
    uint32_t t_start = micros();
    int predicted_class = classifier.predict(kSampleFeatures, 4);
    uint32_t t_elapsed = micros() - t_start;

    Serial.print(F("Predicted Class : ")); Serial.println(predicted_class);
    Serial.print(F("Inference Time  : ")); Serial.print(t_elapsed); Serial.println(F(" us"));

    // Query class probabilities (Scikit-Learn predict_proba)
    const size_t n_classes = classifier.num_classes();
    Serial.println(F("Normalized Probabilities (predict_proba):"));
    for (size_t c = 0; c < n_classes; ++c) {
        float prob = classifier.predict_proba(kSampleFeatures, c);
        Serial.print(F("  - Class [")); Serial.print(c); Serial.print(F("]: "));
        Serial.print(prob * 100.0f, 2); Serial.println(F("%"));
    }
    Serial.println(F("==================================================\n"));
}

void loop() {
    static uint32_t iteration = 0;
    iteration++;

    // Perturb features slightly to simulate live dynamic sensor inputs
    float live_input[4];
    float noise = (float)(iteration % 10) * 0.02f;
    for (int i = 0; i < 4; ++i) {
        live_input[i] = kSampleFeatures[i] + ((i % 2 == 0) ? noise : -noise);
    }

    uint32_t t0 = micros();
    int pred = classifier.predict(live_input, 4);
    uint32_t dt = micros() - t0;
    float confidence = classifier.predict_proba(live_input, pred);

    Serial.print(F("[Stream #")); Serial.print(iteration);
    Serial.print(F("] -> Predicted Class: ")); Serial.print(pred);
    Serial.print(F(" | Confidence: ")); Serial.print(confidence * 100.0f, 1);
    Serial.print(F("% | Latency: ")); Serial.print(dt); Serial.println(F(" us"));

    delay(2000);
}
