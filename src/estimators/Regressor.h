#ifndef TINYISB_REGRESSOR_H
#define TINYISB_REGRESSOR_H

#include "../core/Engine.h"

namespace tinyisb {

/**
 * @brief High-level Scikit-Learn-inspired Regressor for TinyML.
 *
 * Implements Scikit-Learn estimator patterns for regression:
 *  - begin(model_data, arena, arena_size)
 *  - predict(input, len) -> returns scalar float prediction
 *  - predict(input, in_len, output, out_len) -> multi-output vector regression
 *
 * All operations execute with zero dynamic heap allocation during inference.
 */
class Regressor : public Engine {
public:
    Regressor();
    virtual ~Regressor() = default;

    /**
     * @brief Run inference and return a single continuous scalar prediction.
     * @param input Pointer to contiguous float array of input features.
     * @param len Number of elements in the input feature array.
     * @return Continuous scalar prediction value, or NAN on failure.
     */
    float predict(const float* input, size_t len);

    /**
     * @brief Run inference and populate a multi-dimensional continuous output vector.
     * @param input Pointer to input float array.
     * @param in_len Number of elements in input array.
     * @param output Destination array for dequantized output values.
     * @param out_len Capacity of output array.
     * @return Status::OK or error code.
     */
    Status predict(const float* input, size_t in_len, float* output, size_t out_len);

    /**
     * @brief Number of continuous outputs produced by the regression model.
     */
    size_t num_outputs() const {
        return output_elements(0);
    }

    /**
     * @brief Return the scalar value from the most recent predict() call.
     */
    float last_prediction() const {
        return _last_prediction;
    }

private:
    float _last_prediction;
};

} // namespace tinyisb

#endif // TINYISB_REGRESSOR_H
