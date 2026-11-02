#include "Classifier.h"

namespace tinyisb {

Classifier::Classifier()
    : Engine(),
      _last_argmax(-1),
      _last_confidence(0.0f) {}

int Classifier::predict(const float* input, size_t len) {
    if (!_initialized) {
        return -1;
    }

    Status s = set_input(input, len);
    if (s != Status::OK) {
        return -2;
    }

    s = invoke();
    if (s != Status::OK) {
        return -3;
    }

    const size_t n_classes = num_classes();
    if (n_classes == 0) {
        return -4;
    }

    float max_val = -1e30f;
    int best_class = 0;

    for (size_t i = 0; i < n_classes; ++i) {
        float val = get_output_value(i);
        if (val > max_val) {
            max_val = val;
            best_class = static_cast<int>(i);
        }
    }

    _last_argmax = best_class;
    _last_confidence = max_val;
    return best_class;
}

float Classifier::predict_proba(const float* input, size_t class_idx) {
    if (!_initialized) {
        return NAN;
    }

    const size_t n_classes = num_classes();
    if (class_idx >= n_classes) {
        return NAN;
    }

    // Set input and run inference
    Status s = set_input(input, input_elements(0));
    if (s != Status::OK) {
        return NAN;
    }

    s = invoke();
    if (s != Status::OK) {
        return NAN;
    }

    // Inspect outputs to determine if they are already Softmax probabilities or raw logits
    float sum = 0.0f;
    float max_val = -1e30f;
    bool all_non_negative = true;

    for (size_t i = 0; i < n_classes; ++i) {
        float v = get_output_value(i);
        if (v < 0.0f) {
            all_non_negative = false;
        }
        if (v > max_val) {
            max_val = v;
        }
        sum += v;
    }

    // If already normalized probabilities (e.g. model ends with Softmax operator)
    if (all_non_negative && fabsf(sum - 1.0f) < 0.10f) {
        float p = get_output_value(class_idx);
        if (p < 0.0f) p = 0.0f;
        if (p > 1.0f) p = 1.0f;
        return p;
    }

    // Output is raw logits: compute numerically stable Softmax on the fly
    float exp_sum = 0.0f;
    float target_exp = 0.0f;

    for (size_t i = 0; i < n_classes; ++i) {
        float v = get_output_value(i);
        float e = expf(v - max_val);
        exp_sum += e;
        if (i == class_idx) {
            target_exp = e;
        }
    }

    if (exp_sum <= 1e-12f) {
        return 0.0f;
    }

    float proba = target_exp / exp_sum;
    if (proba < 0.0f) proba = 0.0f;
    if (proba > 1.0f) proba = 1.0f;
    return proba;
}

Status Classifier::predict_proba(const float* input, size_t len, float* out_probas, size_t num_classes_len) {
    if (!_initialized) {
        return Status::ERR_NOT_INITIALIZED;
    }
    if (input == nullptr || out_probas == nullptr) {
        return Status::ERR_NULL_POINTER;
    }

    const size_t n_classes = num_classes();
    if (num_classes_len < n_classes) {
        return Status::ERR_OUTPUT_BUFFER_TOO_SMALL;
    }

    Status s = set_input(input, len);
    if (s != Status::OK) {
        return s;
    }

    s = invoke();
    if (s != Status::OK) {
        return s;
    }

    float sum = 0.0f;
    float max_val = -1e30f;
    bool all_non_negative = true;

    for (size_t i = 0; i < n_classes; ++i) {
        float v = get_output_value(i);
        if (v < 0.0f) all_non_negative = false;
        if (v > max_val) max_val = v;
        sum += v;
    }

    if (all_non_negative && fabsf(sum - 1.0f) < 0.10f) {
        for (size_t i = 0; i < n_classes; ++i) {
            float p = get_output_value(i);
            if (p < 0.0f) p = 0.0f;
            if (p > 1.0f) p = 1.0f;
            out_probas[i] = p;
        }
        return Status::OK;
    }

    // Apply numerically stable Softmax
    float exp_sum = 0.0f;
    for (size_t i = 0; i < n_classes; ++i) {
        float v = get_output_value(i);
        float e = expf(v - max_val);
        out_probas[i] = e;
        exp_sum += e;
    }

    const float inv_sum = (exp_sum > 1e-12f) ? (1.0f / exp_sum) : 0.0f;
    for (size_t i = 0; i < n_classes; ++i) {
        out_probas[i] *= inv_sum;
        if (out_probas[i] < 0.0f) out_probas[i] = 0.0f;
        if (out_probas[i] > 1.0f) out_probas[i] = 1.0f;
    }

    return Status::OK;
}

} // namespace tinyisb
