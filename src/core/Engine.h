#ifndef TINYISB_ENGINE_H
#define TINYISB_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "Types.h"

// Platform detection
#if defined(ARDUINO)
    #include <Arduino.h>
#endif

// Include TensorFlow Lite Micro headers with multi-ecosystem fallback
#if !defined(TENSORFLOW_LITE_MICRO_MICRO_INTERPRETER_H_)
    #if __has_include(<tensorflow-runtime-universal.h>)
        #include <tensorflow-runtime-universal.h>
    #elif __has_include(<TensorFlowLite.h>)
        #include <TensorFlowLite.h>
    #elif __has_include(<Chirale_TensorFlowLite.h>)
        #include <Chirale_TensorFlowLite.h>
    #elif __has_include(<MicroTensorFlowLite.h>)
        #include <MicroTensorFlowLite.h>
    #endif
#endif

#if __has_include(<tensorflow/lite/schema/schema_generated.h>)
    #include <tensorflow/lite/schema/schema_generated.h>
    #include <tensorflow/lite/micro/all_ops_resolver.h>
    #include <tensorflow/lite/micro/micro_interpreter.h>
    #include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#endif

#if __has_include(<tensorflow/lite/version.h>)
    #include <tensorflow/lite/version.h>
#endif

#ifndef TFLITE_SCHEMA_VERSION
    #define TFLITE_SCHEMA_VERSION (3)
#endif

namespace tinyisb {

#if defined(ESP32)
namespace internal {
    class UniversalOpResolver : public tflite::MicroMutableOpResolver<64> {
    public:
        UniversalOpResolver() {
            AddFullyConnected();
            AddConv2D();
            AddDepthwiseConv2D();
            AddRelu();
            AddRelu6();
            AddSoftmax();
            AddReshape();
            AddAveragePool2D();
            AddMaxPool2D();
            AddAdd();
            AddSub();
            AddMul();
            AddLogistic();
            AddQuantize();
            AddDequantize();
            AddPack();
            AddUnpack();
            AddPad();
            AddConcatenation();
            AddMean();
            AddMinimum();
            AddMaximum();
            AddStridedSlice();
            AddSplit();
            AddArgMax();
        }
    };
} // namespace internal
using DefaultResolverType = internal::UniversalOpResolver;
#else
using DefaultResolverType = tflite::AllOpsResolver;
#endif

/**
 * @brief Low-level TinyML execution engine wrapping tflite::MicroInterpreter.
 *
 * Designed specifically for bare-metal / RTOS embedded systems:
 *  - Enforces zero dynamic heap allocation during model invocation.
 *  - Requires an externally provided contiguous tensor arena buffer.
 *  - Validates model schema against TFLITE_SCHEMA_VERSION.
 *  - Supports both automatic AllOpsResolver and custom MicroOpResolvers.
 *  - Automatically maps and quantizes/dequantizes float tensors to/from INT8 / UINT8 / FLOAT32.
 */
class Engine {
public:
    Engine();
    virtual ~Engine();

    /**
     * @brief Initialize the model with the default AllOpsResolver.
     * @param model_data Pointer to the raw TFLite model flatbuffer array.
     * @param tensor_arena Pointer to a contiguous RAM buffer dedicated to TFLM activations.
     * @param arena_size Total capacity in bytes of tensor_arena.
     * @return Status::OK on success, or specific error code.
     */
    Status begin(const uint8_t* model_data, uint8_t* tensor_arena, size_t arena_size);

    /**
     * @brief Initialize the model with a custom OpResolver (e.g. MicroMutableOpResolver) to save Flash.
     * @param model_data Pointer to the raw TFLite model flatbuffer array.
     * @param tensor_arena Pointer to contiguous RAM buffer.
     * @param arena_size Capacity of arena in bytes.
     * @param custom_resolver Reference to a configured MicroOpResolver.
     * @return Status::OK on success, or specific error code.
     */
    Status begin(const uint8_t* model_data, uint8_t* tensor_arena, size_t arena_size, tflite::MicroOpResolver& custom_resolver);

    /**
     * @brief Execute model inference (wraps MicroInterpreter::Invoke).
     * @return Status::OK on success, or Status::ERR_INVOKE_FAILED.
     */
    Status invoke();

    /**
     * @brief Copy float input data into the model input tensor, handling quantization automatically.
     * @param data Array of float input values.
     * @param len Number of elements in data.
     * @param tensor_idx Index of the input tensor (default 0).
     * @return Status::OK or error code.
     */
    Status set_input(const float* data, size_t len, size_t tensor_idx = 0);

    /**
     * @brief Copy and dequantize output tensor data into a user float buffer.
     * @param out_data Destination array for float values.
     * @param len Capacity / number of elements to copy.
     * @param tensor_idx Index of the output tensor (default 0).
     * @return Status::OK or error code.
     */
    Status get_output(float* out_data, size_t len, size_t tensor_idx = 0) const;

    /**
     * @brief Get a single dequantized float value from the output tensor.
     * @param elem_idx Element index in the output tensor.
     * @param tensor_idx Output tensor index (default 0).
     * @return Dequantized float value, or NAN if error/uninitialized.
     */
    float get_output_value(size_t elem_idx = 0, size_t tensor_idx = 0) const;

    // --- State and Metadata Inspection ---

    bool is_initialized() const { return _initialized; }
    size_t arena_used_bytes() const;
    size_t arena_total_bytes() const { return _arena_size; }

    TensorShape input_shape(size_t tensor_idx = 0) const;
    TensorShape output_shape(size_t tensor_idx = 0) const;
    size_t input_elements(size_t tensor_idx = 0) const;
    size_t output_elements(size_t tensor_idx = 0) const;
    int input_type(size_t tensor_idx = 0) const;
    int output_type(size_t tensor_idx = 0) const;

    QuantParams input_quantization(size_t tensor_idx = 0) const;
    QuantParams output_quantization(size_t tensor_idx = 0) const;
    ModelMetadata get_metadata() const;

    // --- Direct Tensor Access ---
    TfLiteTensor* input_tensor(size_t tensor_idx = 0);
    const TfLiteTensor* input_tensor(size_t tensor_idx = 0) const;
    TfLiteTensor* output_tensor(size_t tensor_idx = 0);
    const TfLiteTensor* output_tensor(size_t tensor_idx = 0) const;
    tflite::MicroInterpreter* interpreter() { return _interpreter; }
    const tflite::MicroInterpreter* interpreter() const { return _interpreter; }

    /**
     * @brief Print human-readable inspection details to Serial (Arduino) or standard stream.
     */
    void print_metadata() const;

protected:
    Status _init_internal(const uint8_t* model_data, uint8_t* tensor_arena, size_t arena_size, tflite::MicroOpResolver& resolver);
    void _extract_shape(const TfLiteTensor* tensor, TensorShape& shape) const;

    const tflite::Model* _model;
    DefaultResolverType _default_resolver;
    tflite::MicroOpResolver* _resolver;
    tflite::MicroInterpreter* _interpreter;
    TfLiteTensor* _input_tensor;
    TfLiteTensor* _output_tensor;
    uint8_t* _arena;
    size_t _arena_size;
    bool _initialized;
};

} // namespace tinyisb

#endif // TINYISB_ENGINE_H
