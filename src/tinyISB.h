#ifndef TINYISB_H
#define TINYISB_H

/**
 * @file tinyISB.h
 * @brief tinyISB: High-level Scikit-Learn inspired TinyML Inference Library
 *
 * Target Microcontrollers:
 *  - Raspberry Pi Pico (RP2040 Cortex-M0+) & Pico 2 (RP2350 Cortex-M33)
 *  - Espressif ESP32-S3 (Xtensa LX7) & ESP32-C6 (RISC-V)
 *  - Arduino Nano 33 BLE Sense (Nordic nRF52840 Cortex-M4F)
 *
 * Core Guarantees:
 *  1. Zero dynamic heap allocation during inference.
 *  2. Automatic INT8/UINT8 quantization & de-quantization handling.
 *  3. Scikit-Learn API idioms: begin(), predict(), predict_proba().
 *  4. External contiguous tensor arena buffer management.
 */

#define TINYISB_VERSION_MAJOR 1
#define TINYISB_VERSION_MINOR 0
#define TINYISB_VERSION_PATCH 0
#define TINYISB_VERSION_STRING "1.0.0"

#include "core/Types.h"
#include "core/Engine.h"
#include "estimators/Classifier.h"
#include "estimators/Regressor.h"

// Convenience aliases for top-level usage
namespace tinyisb {
    using ModelEngine = Engine;
}

#endif // TINYISB_H
