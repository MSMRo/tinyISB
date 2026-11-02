#include "Regressor.h"

namespace tinyisb {

Regressor::Regressor()
    : Engine(),
      _last_prediction(NAN) {}

float Regressor::predict(const float* input, size_t len) {
    if (!_initialized) {
        return NAN;
    }

    Status s = set_input(input, len);
    if (s != Status::OK) {
        return NAN;
    }

    s = invoke();
    if (s != Status::OK) {
        return NAN;
    }

    float val = get_output_value(0);
    _last_prediction = val;
    return val;
}

Status Regressor::predict(const float* input, size_t in_len, float* output, size_t out_len) {
    if (!_initialized) {
        return Status::ERR_NOT_INITIALIZED;
    }
    if (input == nullptr || output == nullptr) {
        return Status::ERR_NULL_POINTER;
    }

    Status s = set_input(input, in_len);
    if (s != Status::OK) {
        return s;
    }

    s = invoke();
    if (s != Status::OK) {
        return s;
    }

    s = get_output(output, out_len);
    if (s != Status::OK) {
        return s;
    }

    if (out_len > 0) {
        _last_prediction = output[0];
    }

    return Status::OK;
}

} // namespace tinyisb
