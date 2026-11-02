#ifndef TINYISB_TYPES_H
#define TINYISB_TYPES_H

#include <stddef.h>
#include <stdint.h>

namespace tinyisb {

/**
 * @brief Status return codes for tinyISB inference engine operations.
 */
enum class Status : int8_t {
    OK = 0,
    ERR_NULL_POINTER = -1,
    ERR_INVALID_SCHEMA = -2,
    ERR_RESOLVER_FAILED = -3,
    ERR_ARENA_ALLOCATION_FAILED = -4,
    ERR_INVOKE_FAILED = -5,
    ERR_INPUT_DIM_MISMATCH = -6,
    ERR_UNSUPPORTED_DATA_TYPE = -7,
    ERR_NOT_INITIALIZED = -8,
    ERR_INDEX_OUT_OF_BOUNDS = -9,
    ERR_OUTPUT_BUFFER_TOO_SMALL = -10
};

/**
 * @brief Helper to convert Status enum to human-readable string.
 */
inline const char* status_to_string(Status status) {
    switch (status) {
        case Status::OK:
            return "OK";
        case Status::ERR_NULL_POINTER:
            return "ERR_NULL_POINTER: Received null model, arena, or data pointer";
        case Status::ERR_INVALID_SCHEMA:
            return "ERR_INVALID_SCHEMA: Model schema version does not match TFLITE_SCHEMA_VERSION";
        case Status::ERR_RESOLVER_FAILED:
            return "ERR_RESOLVER_FAILED: Operator resolver missing required model operators";
        case Status::ERR_ARENA_ALLOCATION_FAILED:
            return "ERR_ARENA_ALLOCATION_FAILED: Tensor arena insufficient or tensor allocation failed";
        case Status::ERR_INVOKE_FAILED:
            return "ERR_INVOKE_FAILED: MicroInterpreter::Invoke() execution failed";
        case Status::ERR_INPUT_DIM_MISMATCH:
            return "ERR_INPUT_DIM_MISMATCH: Input array length does not match model input tensor";
        case Status::ERR_UNSUPPORTED_DATA_TYPE:
            return "ERR_UNSUPPORTED_DATA_TYPE: Tensor data type not supported (only float32/int8/uint8)";
        case Status::ERR_NOT_INITIALIZED:
            return "ERR_NOT_INITIALIZED: Model engine begin() has not been called or failed";
        case Status::ERR_INDEX_OUT_OF_BOUNDS:
            return "ERR_INDEX_OUT_OF_BOUNDS: Requested class index or tensor index is out of bounds";
        case Status::ERR_OUTPUT_BUFFER_TOO_SMALL:
            return "ERR_OUTPUT_BUFFER_TOO_SMALL: User output buffer smaller than model output tensor";
        default:
            return "ERR_UNKNOWN: Unspecified tinyISB runtime error";
    }
}

/**
 * @brief Tensor dimension and shape descriptor.
 */
struct TensorShape {
    int dims[6];
    size_t num_dims;
    size_t total_elements;

    TensorShape() : num_dims(0), total_elements(0) {
        for (size_t i = 0; i < 6; ++i) {
            dims[i] = 0;
        }
    }
};

/**
 * @brief Quantization parameters (scale and zero-point).
 */
struct QuantParams {
    float scale;
    int32_t zero_point;
    bool is_quantized;

    QuantParams() : scale(1.0f), zero_point(0), is_quantized(false) {}
};

/**
 * @brief Structural metadata inspected directly from the loaded TFLite model.
 */
struct ModelMetadata {
    TensorShape input_shape;
    TensorShape output_shape;
    int input_type;       // Corresponds to TfLiteType
    int output_type;      // Corresponds to TfLiteType
    QuantParams input_quant;
    QuantParams output_quant;
    size_t arena_used_bytes;
    int model_version;

    ModelMetadata()
        : input_type(0), output_type(0), arena_used_bytes(0), model_version(0) {}
};

} // namespace tinyisb

#endif // TINYISB_TYPES_H
