#include "Engine.h"

#if !defined(ARDUINO)
#include <stdio.h>
#endif

namespace tinyisb {

Engine::Engine()
    : _model(nullptr),
      _resolver(nullptr),
      _interpreter(nullptr),
      _input_tensor(nullptr),
      _output_tensor(nullptr),
      _arena(nullptr),
      _arena_size(0),
      _initialized(false) {}

Engine::~Engine() {
    if (_interpreter != nullptr) {
        delete _interpreter;
        _interpreter = nullptr;
    }
    _initialized = false;
}

Status Engine::begin(const uint8_t* model_data, uint8_t* tensor_arena, size_t arena_size) {
    if (model_data == nullptr || tensor_arena == nullptr || arena_size == 0) {
        return Status::ERR_NULL_POINTER;
    }

    _resolver = &_default_resolver;
    return _init_internal(model_data, tensor_arena, arena_size, *_resolver);
}

Status Engine::begin(const uint8_t* model_data, uint8_t* tensor_arena, size_t arena_size, tflite::MicroOpResolver& custom_resolver) {
    if (model_data == nullptr || tensor_arena == nullptr || arena_size == 0) {
        return Status::ERR_NULL_POINTER;
    }

    _resolver = &custom_resolver;
    return _init_internal(model_data, tensor_arena, arena_size, *_resolver);
}

Status Engine::_init_internal(const uint8_t* model_data, uint8_t* tensor_arena, size_t arena_size, tflite::MicroOpResolver& resolver) {
    _arena = tensor_arena;
    _arena_size = arena_size;
    _initialized = false;

    // Validate model schema version
    _model = tflite::GetModel(model_data);
    if (_model == nullptr) {
        return Status::ERR_NULL_POINTER;
    }

    if (_model->version() != TFLITE_SCHEMA_VERSION) {
        return Status::ERR_INVALID_SCHEMA;
    }

    // Free existing interpreter if re-initializing
    if (_interpreter != nullptr) {
        delete _interpreter;
        _interpreter = nullptr;
    }

    // Instantiate micro interpreter
    _interpreter = new tflite::MicroInterpreter(_model, resolver, _arena, _arena_size);
    if (_interpreter == nullptr) {
        return Status::ERR_ARENA_ALLOCATION_FAILED;
    }

    // Allocate tensors from the arena buffer
    TfLiteStatus alloc_status = _interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        delete _interpreter;
        _interpreter = nullptr;
        return Status::ERR_ARENA_ALLOCATION_FAILED;
    }

    // Cache primary input and output tensors
    _input_tensor = _interpreter->input(0);
    _output_tensor = _interpreter->output(0);

    if (_input_tensor == nullptr || _output_tensor == nullptr) {
        return Status::ERR_ARENA_ALLOCATION_FAILED;
    }

    _initialized = true;
    return Status::OK;
}

Status Engine::invoke() {
    if (!_initialized || _interpreter == nullptr) {
        return Status::ERR_NOT_INITIALIZED;
    }

    TfLiteStatus status = _interpreter->Invoke();
    if (status != kTfLiteOk) {
        return Status::ERR_INVOKE_FAILED;
    }

    return Status::OK;
}

Status Engine::set_input(const float* data, size_t len, size_t tensor_idx) {
    if (!_initialized || _interpreter == nullptr) {
        return Status::ERR_NOT_INITIALIZED;
    }
    if (data == nullptr) {
        return Status::ERR_NULL_POINTER;
    }

    TfLiteTensor* in_tensor = (tensor_idx == 0) ? _input_tensor : _interpreter->input(tensor_idx);
    if (in_tensor == nullptr) {
        return Status::ERR_INDEX_OUT_OF_BOUNDS;
    }

    TensorShape shape;
    _extract_shape(in_tensor, shape);
    if (len != shape.total_elements) {
        return Status::ERR_INPUT_DIM_MISMATCH;
    }

    switch (in_tensor->type) {
        case kTfLiteFloat32: {
            float* dst = in_tensor->data.f;
            for (size_t i = 0; i < len; ++i) {
                dst[i] = data[i];
            }
            break;
        }
        case kTfLiteInt8: {
            const float scale = in_tensor->params.scale;
            const int32_t zero_point = in_tensor->params.zero_point;
            int8_t* dst = in_tensor->data.int8;

            // Handle scale = 0 edge case safely
            const float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 1.0f;
            for (size_t i = 0; i < len; ++i) {
                int32_t q = static_cast<int32_t>(roundf(data[i] * inv_scale)) + zero_point;
                if (q < -128) q = -128;
                else if (q > 127) q = 127;
                dst[i] = static_cast<int8_t>(q);
            }
            break;
        }
        case kTfLiteUInt8: {
            const float scale = in_tensor->params.scale;
            const int32_t zero_point = in_tensor->params.zero_point;
            uint8_t* dst = in_tensor->data.uint8;

            const float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 1.0f;
            for (size_t i = 0; i < len; ++i) {
                int32_t q = static_cast<int32_t>(roundf(data[i] * inv_scale)) + zero_point;
                if (q < 0) q = 0;
                else if (q > 255) q = 255;
                dst[i] = static_cast<uint8_t>(q);
            }
            break;
        }
        default:
            return Status::ERR_UNSUPPORTED_DATA_TYPE;
    }

    return Status::OK;
}

Status Engine::get_output(float* out_data, size_t len, size_t tensor_idx) const {
    if (!_initialized || _interpreter == nullptr) {
        return Status::ERR_NOT_INITIALIZED;
    }
    if (out_data == nullptr) {
        return Status::ERR_NULL_POINTER;
    }

    const TfLiteTensor* out_tensor = (tensor_idx == 0) ? _output_tensor : _interpreter->output(tensor_idx);
    if (out_tensor == nullptr) {
        return Status::ERR_INDEX_OUT_OF_BOUNDS;
    }

    TensorShape shape;
    _extract_shape(out_tensor, shape);
    if (len < shape.total_elements) {
        return Status::ERR_OUTPUT_BUFFER_TOO_SMALL;
    }

    const size_t num_elems = shape.total_elements;
    switch (out_tensor->type) {
        case kTfLiteFloat32: {
            const float* src = out_tensor->data.f;
            for (size_t i = 0; i < num_elems; ++i) {
                out_data[i] = src[i];
            }
            break;
        }
        case kTfLiteInt8: {
            const float scale = out_tensor->params.scale;
            const int32_t zero_point = out_tensor->params.zero_point;
            const int8_t* src = out_tensor->data.int8;
            for (size_t i = 0; i < num_elems; ++i) {
                out_data[i] = (static_cast<float>(src[i]) - static_cast<float>(zero_point)) * scale;
            }
            break;
        }
        case kTfLiteUInt8: {
            const float scale = out_tensor->params.scale;
            const int32_t zero_point = out_tensor->params.zero_point;
            const uint8_t* src = out_tensor->data.uint8;
            for (size_t i = 0; i < num_elems; ++i) {
                out_data[i] = (static_cast<float>(src[i]) - static_cast<float>(zero_point)) * scale;
            }
            break;
        }
        default:
            return Status::ERR_UNSUPPORTED_DATA_TYPE;
    }

    return Status::OK;
}

float Engine::get_output_value(size_t elem_idx, size_t tensor_idx) const {
    if (!_initialized || _interpreter == nullptr) {
        return NAN;
    }

    const TfLiteTensor* out_tensor = (tensor_idx == 0) ? _output_tensor : _interpreter->output(tensor_idx);
    if (out_tensor == nullptr) {
        return NAN;
    }

    TensorShape shape;
    _extract_shape(out_tensor, shape);
    if (elem_idx >= shape.total_elements) {
        return NAN;
    }

    switch (out_tensor->type) {
        case kTfLiteFloat32:
            return out_tensor->data.f[elem_idx];
        case kTfLiteInt8: {
            const float scale = out_tensor->params.scale;
            const int32_t zero_point = out_tensor->params.zero_point;
            return (static_cast<float>(out_tensor->data.int8[elem_idx]) - static_cast<float>(zero_point)) * scale;
        }
        case kTfLiteUInt8: {
            const float scale = out_tensor->params.scale;
            const int32_t zero_point = out_tensor->params.zero_point;
            return (static_cast<float>(out_tensor->data.uint8[elem_idx]) - static_cast<float>(zero_point)) * scale;
        }
        default:
            return NAN;
    }
}

size_t Engine::arena_used_bytes() const {
    if (!_initialized || _interpreter == nullptr) {
        return 0;
    }
    return _interpreter->arena_used_bytes();
}

TensorShape Engine::input_shape(size_t tensor_idx) const {
    TensorShape shape;
    const TfLiteTensor* tensor = input_tensor(tensor_idx);
    if (tensor != nullptr) {
        _extract_shape(tensor, shape);
    }
    return shape;
}

TensorShape Engine::output_shape(size_t tensor_idx) const {
    TensorShape shape;
    const TfLiteTensor* tensor = output_tensor(tensor_idx);
    if (tensor != nullptr) {
        _extract_shape(tensor, shape);
    }
    return shape;
}

size_t Engine::input_elements(size_t tensor_idx) const {
    return input_shape(tensor_idx).total_elements;
}

size_t Engine::output_elements(size_t tensor_idx) const {
    return output_shape(tensor_idx).total_elements;
}

int Engine::input_type(size_t tensor_idx) const {
    const TfLiteTensor* tensor = input_tensor(tensor_idx);
    return (tensor != nullptr) ? static_cast<int>(tensor->type) : -1;
}

int Engine::output_type(size_t tensor_idx) const {
    const TfLiteTensor* tensor = output_tensor(tensor_idx);
    return (tensor != nullptr) ? static_cast<int>(tensor->type) : -1;
}

QuantParams Engine::input_quantization(size_t tensor_idx) const {
    QuantParams qp;
    const TfLiteTensor* tensor = input_tensor(tensor_idx);
    if (tensor != nullptr) {
        qp.scale = tensor->params.scale;
        qp.zero_point = tensor->params.zero_point;
        qp.is_quantized = (tensor->type == kTfLiteInt8 || tensor->type == kTfLiteUInt8);
    }
    return qp;
}

QuantParams Engine::output_quantization(size_t tensor_idx) const {
    QuantParams qp;
    const TfLiteTensor* tensor = output_tensor(tensor_idx);
    if (tensor != nullptr) {
        qp.scale = tensor->params.scale;
        qp.zero_point = tensor->params.zero_point;
        qp.is_quantized = (tensor->type == kTfLiteInt8 || tensor->type == kTfLiteUInt8);
    }
    return qp;
}

ModelMetadata Engine::get_metadata() const {
    ModelMetadata meta;
    if (!_initialized || _interpreter == nullptr) {
        return meta;
    }
    meta.input_shape = input_shape(0);
    meta.output_shape = output_shape(0);
    meta.input_type = input_type(0);
    meta.output_type = output_type(0);
    meta.input_quant = input_quantization(0);
    meta.output_quant = output_quantization(0);
    meta.arena_used_bytes = arena_used_bytes();
    meta.model_version = (_model != nullptr) ? _model->version() : 0;
    return meta;
}

TfLiteTensor* Engine::input_tensor(size_t tensor_idx) {
    if (!_initialized || _interpreter == nullptr) return nullptr;
    return (tensor_idx == 0) ? _input_tensor : _interpreter->input(tensor_idx);
}

const TfLiteTensor* Engine::input_tensor(size_t tensor_idx) const {
    if (!_initialized || _interpreter == nullptr) return nullptr;
    return (tensor_idx == 0) ? _input_tensor : _interpreter->input(tensor_idx);
}

TfLiteTensor* Engine::output_tensor(size_t tensor_idx) {
    if (!_initialized || _interpreter == nullptr) return nullptr;
    return (tensor_idx == 0) ? _output_tensor : _interpreter->output(tensor_idx);
}

const TfLiteTensor* Engine::output_tensor(size_t tensor_idx) const {
    if (!_initialized || _interpreter == nullptr) return nullptr;
    return (tensor_idx == 0) ? _output_tensor : _interpreter->output(tensor_idx);
}

void Engine::_extract_shape(const TfLiteTensor* tensor, TensorShape& shape) const {
    if (tensor == nullptr || tensor->dims == nullptr) {
        shape.num_dims = 0;
        shape.total_elements = 0;
        return;
    }

    shape.num_dims = tensor->dims->size;
    if (shape.num_dims > 6) {
        shape.num_dims = 6;
    }

    size_t total = 1;
    for (size_t i = 0; i < shape.num_dims; ++i) {
        shape.dims[i] = tensor->dims->data[i];
        if (shape.dims[i] > 0) {
            total *= static_cast<size_t>(shape.dims[i]);
        }
    }
    shape.total_elements = total;
}

void Engine::print_metadata() const {
#if defined(ARDUINO)
    if (!_initialized) {
        Serial.println(F("[tinyISB] Model Engine is NOT initialized!"));
        return;
    }

    ModelMetadata meta = get_metadata();
    Serial.println(F("========================================"));
    Serial.println(F("           tinyISB Model Metadata       "));
    Serial.println(F("========================================"));
    Serial.print(F("Schema Version   : ")); Serial.println(meta.model_version);
    Serial.print(F("Arena Used Bytes : ")); Serial.print(meta.arena_used_bytes);
    Serial.print(F(" / ")); Serial.print(_arena_size);
    Serial.print(F(" ("));
    Serial.print((meta.arena_used_bytes * 100) / (_arena_size > 0 ? _arena_size : 1));
    Serial.println(F("%)"));

    Serial.print(F("Input Dims (")); Serial.print(meta.input_shape.num_dims); Serial.print(F("): ["));
    for (size_t i = 0; i < meta.input_shape.num_dims; ++i) {
        Serial.print(meta.input_shape.dims[i]);
        if (i + 1 < meta.input_shape.num_dims) Serial.print(F(", "));
    }
    Serial.print(F("] Total: ")); Serial.println(meta.input_shape.total_elements);

    Serial.print(F("Input Quantized  : "));
    Serial.print(meta.input_quant.is_quantized ? F("YES (INT8)") : F("NO (Float32)"));
    if (meta.input_quant.is_quantized) {
        Serial.print(F(" | Scale=")); Serial.print(meta.input_quant.scale, 6);
        Serial.print(F(" ZeroPoint=")); Serial.print(meta.input_quant.zero_point);
    }
    Serial.println();

    Serial.print(F("Output Dims (")); Serial.print(meta.output_shape.num_dims); Serial.print(F("): ["));
    for (size_t i = 0; i < meta.output_shape.num_dims; ++i) {
        Serial.print(meta.output_shape.dims[i]);
        if (i + 1 < meta.output_shape.num_dims) Serial.print(F(", "));
    }
    Serial.print(F("] Total: ")); Serial.println(meta.output_shape.total_elements);

    Serial.print(F("Output Quantized : "));
    Serial.print(meta.output_quant.is_quantized ? F("YES (INT8)") : F("NO (Float32)"));
    if (meta.output_quant.is_quantized) {
        Serial.print(F(" | Scale=")); Serial.print(meta.output_quant.scale, 6);
        Serial.print(F(" ZeroPoint=")); Serial.print(meta.output_quant.zero_point);
    }
    Serial.println();
    Serial.println(F("========================================"));
#else
    if (!_initialized) {
        printf("[tinyISB] Model Engine is NOT initialized!\n");
        return;
    }
    ModelMetadata meta = get_metadata();
    printf("========================================\n");
    printf("           tinyISB Model Metadata       \n");
    printf("========================================\n");
    printf("Schema Version   : %d\n", meta.model_version);
    printf("Arena Used Bytes : %zu / %zu\n", meta.arena_used_bytes, _arena_size);
    printf("Input Elements   : %zu (Quantized: %s)\n", meta.input_shape.total_elements, meta.input_quant.is_quantized ? "YES" : "NO");
    printf("Output Elements  : %zu (Quantized: %s)\n", meta.output_shape.total_elements, meta.output_quant.is_quantized ? "YES" : "NO");
    printf("========================================\n");
#endif
}

} // namespace tinyisb
