#ifndef TINYISB_CLASSIFIER_H
#define TINYISB_CLASSIFIER_H

#include "../core/Engine.h"

namespace tinyisb {

/**
 * @brief High-level Scikit-Learn-inspired Classifier for TinyML.
 *
 * Implements Scikit-Learn estimator patterns:
 *  - begin(model_data, arena, arena_size)
 *  - predict(input, len) -> returns argmax class index
 *  - predict_proba(input, class_idx) -> returns normalized confidence [0.0, 1.0]
 *  - predict_proba(input, len, out_probas, num_classes) -> fills full probability array
 *
 * All operations execute with zero dynamic heap allocation.
 */
class Classifier : public Engine {
public:
    Classifier();
    virtual ~Classifier() = default;

    /**
     * @brief Run inference and return the predicted class index (argmax).
     * @param input Pointer to contiguous float array of input features.
     * @param len Number of elements in the input feature array.
     * @return Predicted class index (0 to num_classes - 1), or negative integer on failure.
     */
    int predict(const float* input, size_t len);

    /**
     * @brief Run inference and return normalized confidence for a specific class.
     * @param input Pointer to contiguous float array of input features.
     * @param class_idx Index of the target class.
     * @return Probability/confidence in range [0.0, 1.0], or NAN on error.
     */
    float predict_proba(const float* input, size_t class_idx);

    /**
     * @brief Run inference and populate all class probabilities in user-provided buffer.
     * @param input Pointer to float feature array.
     * @param len Length of feature array.
     * @param out_probas User-provided array of size at least num_classes().
     * @param num_classes Capacity of out_probas.
     * @return Status::OK or error code.
     */
    Status predict_proba(const float* input, size_t len, float* out_probas, size_t num_classes);

    /**
     * @brief Number of classes predicted by the model (size of primary output tensor).
     */
    size_t num_classes() const {
        return output_elements(0);
    }

    /**
     * @brief Return the class index from the most recent predict() call.
     */
    int last_prediction() const {
        return _last_argmax;
    }

    /**
     * @brief Return the confidence score from the most recent predict() call.
     */
    float last_confidence() const {
        return _last_confidence;
    }

private:
    int _last_argmax;
    float _last_confidence;
};

} // namespace tinyisb

#endif // TINYISB_CLASSIFIER_H
