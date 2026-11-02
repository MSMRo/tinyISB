/**
 * @file SignalRegressor.ino
 * @brief Continuous Signal Regression Example using tinyISB
 *
 * Demonstrates:
 *  - High-level tinyisb::Regressor estimator.
 *  - Predicting continuous scalar outputs (e.g. respiration rate in breaths/min).
 *  - Automatic INT8 de-quantization to continuous float.
 *  - Zero dynamic heap allocation during inference.
 *
 * Target Boards:
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
#include "regressor_model_data.h"

// Static 16-byte aligned RAM buffer for activations
constexpr size_t kTensorArenaSize = 4 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// tinyISB Regressor estimator
static tinyisb::Regressor regressor;

// Sample 6-feature test vector: expected output ~ 17.2 to 17.7
static const float kSampleInputs[6] = {-1.2374f, 0.1094f, 1.3286f, 0.3132f, -0.6065f, 0.4559f};

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}

    Serial.println();
    Serial.println(F("=============================================================="));
    Serial.println(F("      tinyISB: Continuous Bio-Signal Regressor                "));
    Serial.println(F("=============================================================="));

    // Initialize Regressor model
    Serial.println(F("[tinyISB] Initializing Regressor model..."));
    tinyisb::Status status = regressor.begin(regressor_model_data, tensor_arena, kTensorArenaSize);

    if (status != tinyisb::Status::OK) {
        Serial.print(F("[FATAL] Regressor initialization failed: "));
        Serial.println(tinyisb::status_to_string(status));
        while (true) { delay(1000); }
    }

    Serial.println(F("[tinyISB] Regressor initialized successfully!"));
    regressor.print_metadata();

    Serial.println(F("\n--- Running Single Test Vector Prediction ---"));
    Serial.print(F("Input Features: ["));
    for (size_t i = 0; i < 6; ++i) {
        Serial.print(kSampleInputs[i], 4);
        if (i < 5) Serial.print(F(", "));
    }
    Serial.println(F("]"));

    uint32_t t_start = micros();
    float predicted_val = regressor.predict(kSampleInputs, 6);
    uint32_t t_inference = micros() - t_start;

    Serial.print(F("Predicted Value  : ")); Serial.print(predicted_val, 4);
    Serial.println(F(" (e.g. breaths/min or Force Units)"));
    Serial.print(F("Inference Time   : ")); Serial.print(t_inference); Serial.println(F(" us"));
    Serial.println(F("==============================================================\n"));
}

void loop() {
    static uint32_t step = 0;
    step++;

    float dynamic_features[6];
    float modulation = sinf((float)step * 0.1f) * 0.3f;
    for (size_t i = 0; i < 6; ++i) {
        dynamic_features[i] = kSampleInputs[i] + modulation * (float)(i + 1) * 0.2f;
    }

    uint32_t t0 = micros();
    float output_val = regressor.predict(dynamic_features, 6);
    uint32_t dt = micros() - t0;

    Serial.print(F("[Sample #")); Serial.print(step);
    Serial.print(F("] Continuous Prediction: ")); Serial.print(output_val, 3);
    Serial.print(F(" | Latency: ")); Serial.print(dt); Serial.println(F(" us"));

    delay(2000);
}
