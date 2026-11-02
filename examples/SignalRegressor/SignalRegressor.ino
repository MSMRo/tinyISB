/**
 * @file SignalRegressor.ino
 * @brief Continuous Signal Regression Example using tinyISB
 *
 * ============================================================================
 * tinyISB: TinyML Scikit-Learn Inference Engine for Microcontrollers
 * ============================================================================
 *
 * Continuous Parameter Regression Walkthrough:
 *  - Domain: Real-time physiological telemetry & biomechanical parameter estimation.
 *  - Problem Formulation: Classification models output discrete categories, but many
 *    embedded applications require predicting continuous physical metrics (e.g. continuous
 *    respiration rate in Breaths Per Minute, muscular exertion force in Newtons, or joint angle).
 *  - Input Feature Vector (6 continuous dimensions):
 *      F1: Mean Absolute Value (MAV)
 *      F2: Root Mean Square (RMS)
 *      F3: Zero-Crossing Rate (ZCR)
 *      F4: Waveform Length (WL)
 *      F5: Spectral Centroid (SC)
 *      F6: Autoregressive Coefficient (AR)
 *  - Automatic De-quantization:
 *      * Model is quantized to INT8 for minimal Flash footprint (~2.5 KB).
 *      * tinyisb::Regressor automatically applies the linear inverse transformation:
 *        y_hat = (q_out - zero_point_out) * scale_out
 *      * Returns a real-world continuous float32 directly without manual bit-shifting!
 *
 * Target Boards:
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

// Include the 16-byte aligned INT8 quantized regressor model header
#include "regressor_model_data.h"

// ============================================================================
// 2. Static Tensor Arena Definition (Zero Dynamic Memory Allocation)
// ============================================================================
// Static 16-byte aligned RAM buffer for regressor activations.
// 4 KB is ample for this compact 3-layer regression MLP.
constexpr size_t kTensorArenaSize = 4 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// ============================================================================
// 3. Estimator Instantiation
// ============================================================================
// High-level tinyisb::Regressor estimator for continuous scalar and vector outputs.
static tinyisb::Regressor regressor;

// Sample 6-feature test vector derived from bio-signal feature extraction.
// Ground truth expected output: ~17.2 to 17.7 Breaths/Min (Normal adult resting rate).
static const float kSampleInputs[6] = {-1.2374f, 0.1094f, 1.3286f, 0.3132f, -0.6065f, 0.4559f};

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}

    Serial.println();
    Serial.println(F("=============================================================="));
    Serial.println(F("      tinyISB: Continuous Bio-Signal Regressor                "));
    Serial.println(F("=============================================================="));

    // ========================================================================
    // 4. Regressor Initialization & Schema Verification
    // ========================================================================
    // regressor.begin() validates FlatBuffers schema, registers operators,
    // verifies that output dimension is valid for regression, and stores
    // quantization scale and zero-point parameters.
    Serial.println(F("[tinyISB] Initializing Regressor model..."));
    tinyisb::Status status = regressor.begin(regressor_model_data, tensor_arena, kTensorArenaSize);

    if (status != tinyisb::Status::OK) {
        Serial.print(F("[FATAL] Regressor initialization failed: "));
        Serial.println(tinyisb::status_to_string(status));
        Serial.println(F("[FATAL] Verify arena size or model alignment."));
        while (true) { delay(1000); }
    }

    Serial.println(F("[tinyISB] Regressor initialized successfully!"));

    // Print model metadata (Input shape [1, 6], Output shape [1, 1], INT8 quantization scale/offset)
    regressor.print_metadata();

    // ========================================================================
    // 5. Benchmark Single Test Feature Vector Prediction
    // ========================================================================
    Serial.println(F("\n--- Running Single Test Vector Prediction ---"));
    Serial.print(F("Input Features: ["));
    for (size_t i = 0; i < 6; ++i) {
        Serial.print(kSampleInputs[i], 4);
        if (i < 5) Serial.print(F(", "));
    }
    Serial.println(F("]"));

    // Measure regression execution latency
    uint32_t t_start = micros();

    // regressor.predict(inputs, length):
    //  - Automatically quantizes 6 float inputs to INT8
    //  - Executes forward inference via tflite::MicroInterpreter::Invoke()
    //  - Reads INT8 output accumulator and automatically converts to float:
    //    y = (q - zero_point) * scale
    //  - Returns continuous float32 scalar
    float predicted_val = regressor.predict(kSampleInputs, 6);
    uint32_t t_inference = micros() - t_start;

    Serial.print(F("Predicted Value  : ")); Serial.print(predicted_val, 4);
    Serial.println(F(" Breaths/Min"));
    Serial.print(F("Inference Time   : ")); Serial.print(t_inference); Serial.println(F(" us"));
    Serial.println(F("==============================================================\n"));
}

void loop() {
    static uint32_t step = 0;
    step++;

    // ========================================================================
    // 6. Dynamic Streaming Ingestion & Real-Time Monitoring
    // ========================================================================
    // Modulate features with a sinusoidal drift to simulate physiological fluctuations
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
    Serial.print(F(" BPM | Latency: ")); Serial.print(dt); Serial.println(F(" us"));

    // Real-time Clinical Alert Monitoring:
    // Normal adult resting respiration: 12-20 Breaths Per Minute
    if (output_val < 10.0f) {
        Serial.println(F("  >>> [WARNING] Bradypnea / Low Respiration Detected! <<<"));
    } else if (output_val > 24.0f) {
        Serial.println(F("  >>> [WARNING] Tachypnea / High Respiration Detected! <<<"));
    }

    delay(2000);
}
