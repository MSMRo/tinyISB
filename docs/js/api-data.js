/**
 * tinyISB API Reference Database
 * Comprehensive C++ API data structured like ARM CMSIS-DSP (arm_math) reference.
 */

const TINYISB_API_DATA = {
  version: "1.0.0",
  lastUpdated: "November 2026",
  license: "MIT",
  modules: [
    {
      id: "estimators",
      name: "Estimators API (Scikit-Learn)",
      description: "High-level ML estimator interfaces inspired by Scikit-Learn for edge microcontrollers.",
      classes: ["Classifier", "Regressor"]
    },
    {
      id: "core",
      name: "Core Engine",
      description: "Low-level TensorFlow Lite Micro inference wrapper and tensor memory manager.",
      classes: ["Engine"]
    },
    {
      id: "types",
      name: "Types & Enumerations",
      description: "Return status codes, shape dimensions, metadata, and quantization parameters.",
      classes: ["Status", "TensorShape", "QuantParams", "ModelMetadata"]
    }
  ],

  classes: {
    Classifier: {
      name: "tinyisb::Classifier",
      category: "Estimators API",
      header: "#include <tinyISB.h>",
      brief: "High-level multi-class classifier providing Scikit-Learn style predict() and predict_proba().",
      description: "The Classifier estimator encapsulates TensorFlow Lite Micro forward inference for categorical classification tasks. It abstracts tensor memory allocation, automatic float-to-INT8 quantization, raw forward inference execution, and post-inference argmax reduction and Softmax probability distribution extraction without any dynamic heap allocation.",
      methods: [
        {
          id: "classifier-begin",
          name: "begin",
          signature: "tinyisb::Status begin(const uint8_t* model_data, uint8_t* tensor_arena, size_t arena_size, tflite::MicroOpResolver* custom_resolver = nullptr);",
          brief: "Initialises the classifier with external model data and a static Tensor Arena.",
          details: "Validates the FlatBuffers model schema version via tflite::GetModel(). If valid, links operators using the platform resolver (UniversalOpResolver on ESP32 or AllOpsResolver on ARM Cortex-M), instantiates tflite::MicroInterpreter, and allocates activation tensors inside the caller-provided static tensor_arena. Inspects input/output shapes and extracts quantization scales and zero-points.",
          params: [
            {
              name: "model_data",
              type: "const uint8_t*",
              direction: "[in]",
              description: "Pointer to the 16-byte aligned .tflite C-array stored in Flash ROM.",
              constraints: "Must not be nullptr. Must be aligned to 16 bytes (alignas(16))."
            },
            {
              name: "tensor_arena",
              type: "uint8_t*",
              direction: "[in,out]",
              description: "Pointer to contiguous pre-allocated static RAM buffer for activations.",
              constraints: "Must not be nullptr. Must be aligned to 16 bytes (alignas(16))."
            },
            {
              name: "arena_size",
              type: "size_t",
              direction: "[in]",
              description: "Total capacity of the tensor_arena buffer in bytes.",
              constraints: "Must be large enough to accommodate model activations (typically 4 KB to 16 KB)."
            },
            {
              name: "custom_resolver",
              type: "tflite::MicroOpResolver*",
              direction: "[in]",
              description: "Optional custom op resolver (e.g. MicroMutableOpResolver<N>) to minimize Flash consumption.",
              constraints: "Defaults to nullptr (uses built-in platform resolver)."
            }
          ],
          returns: {
            type: "tinyisb::Status",
            description: "tinyisb::Status::OK (0) upon successful initialization, or an error status code.",
            errorValues: [
              "Status::ERROR_INVALID_ARGUMENT: model_data or tensor_arena is null, or arena_size is 0.",
              "Status::ERROR_SCHEMA_MISMATCH: Model flatbuffer schema does not match TFLM runtime.",
              "Status::ERROR_ALLOCATION_FAILED: tensor_arena is too small for activation tensors."
            ]
          },
          complexity: {
            heapAlloc: "0 bytes (Deterministic static RAM only)",
            timeComplexity: "O(K) where K is the number of operators during setup",
            stackFootprint: "< 256 bytes"
          },
          preconditions: [
            "Hardware clock and power management must be initialized prior to calling begin().",
            "tensor_arena and model_data must satisfy alignas(16)."
          ],
          example: `constexpr size_t kArenaSize = 4 * 1024; // 4 KB static RAM
alignas(16) static uint8_t tensor_arena[kArenaSize];
static tinyisb::Classifier classifier;

tinyisb::Status status = classifier.begin(model_data, tensor_arena, kArenaSize);
if (status != tinyisb::Status::OK) {
    Serial.print("Init failed: ");
    Serial.println(tinyisb::status_to_string(status));
    while(1);
}`
        },
        {
          id: "classifier-predict",
          name: "predict",
          signature: "int predict(const float* input, size_t len);",
          brief: "Executes model inference and returns the predicted class index (argmax).",
          details: "Quantizes the incoming floating-point feature array according to the model's input tensor scale and zero-point parameters. Copies the quantized values into the model's input tensor, invokes the neural network forward pass, dequantizes the output logits, and returns the index of the class with maximum activation (argmax).",
          params: [
            {
              name: "input",
              type: "const float*",
              direction: "[in]",
              description: "Pointer to the input floating-point feature array.",
              constraints: "Must not be nullptr. Must contain at least 'len' valid floats."
            },
            {
              name: "len",
              type: "size_t",
              direction: "[in]",
              description: "Number of elements in the input feature array.",
              constraints: "Must match input_shape().flat_size()."
            }
          ],
          returns: {
            type: "int",
            description: "Predicted class index in range [0, num_classes() - 1], or -1 on error.",
            errorValues: [
              "-1: Model not initialized, input is null, or length mismatch."
            ]
          },
          complexity: {
            heapAlloc: "0 bytes (Guaranteed zero heap fragmentation)",
            timeComplexity: "O(M) forward pass MAC operations",
            stackFootprint: "< 128 bytes"
          },
          preconditions: [
            "classifier.begin() must have returned tinyisb::Status::OK.",
            "len must exactly match the model's input tensor flat size."
          ],
          example: `float sensory_features[4] = {0.9375f, -0.3574f, 0.3241f, 0.4415f};
int predicted_class = classifier.predict(sensory_features, 4);
Serial.print("Predicted Class: ");
Serial.println(predicted_class);`
        },
        {
          id: "classifier-predict-proba",
          name: "predict_proba",
          signature: "float predict_proba(const float* input, size_t class_idx);",
          brief: "Returns normalized confidence probability [0.0, 1.0] for a specific class.",
          details: "Runs forward inference (if not already cached) and computes a numerically stable Softmax distribution across all output classes: P(c) = exp(z_c - z_max) / sum(exp(z_i - z_max)). Returns the probability for class_idx.",
          params: [
            {
              name: "input",
              type: "const float*",
              direction: "[in]",
              description: "Pointer to the input floating-point feature array.",
              constraints: "Must not be nullptr. Must contain at least the required number of features."
            },
            {
              name: "class_idx",
              type: "size_t",
              direction: "[in]",
              description: "Zero-based index of the target class to query.",
              constraints: "Must be strictly less than num_classes()."
            }
          ],
          returns: {
            type: "float",
            description: "Normalized probability in range [0.0f, 1.0f], or 0.0f on error.",
            errorValues: [
              "0.0f: Invalid class_idx, uninitialized model, or null input."
            ]
          },
          complexity: {
            heapAlloc: "0 bytes",
            timeComplexity: "O(C) Softmax reduction where C is the number of classes",
            stackFootprint: "< 128 bytes"
          },
          preconditions: [
            "class_idx < num_classes()."
          ],
          example: `float conf = classifier.predict_proba(sensory_features, predicted_class);
Serial.print("Confidence: ");
Serial.print(conf * 100.0f);
Serial.println("%");`
        },
        {
          id: "classifier-predict-proba-all",
          name: "predict_proba (all classes)",
          signature: "tinyisb::Status predict_proba(const float* input, float* out_probabilities, size_t len);",
          brief: "Populates an array with normalized probabilities across all classes.",
          details: "Executes forward inference, calculates numerically stable Softmax normalization across the full output tensor, and writes the resulting probability distribution into out_probabilities.",
          params: [
            {
              name: "input",
              type: "const float*",
              direction: "[in]",
              description: "Pointer to the input floating-point feature array.",
              constraints: "Must not be nullptr."
            },
            {
              name: "out_probabilities",
              type: "float*",
              direction: "[out]",
              description: "Destination array to store probability values.",
              constraints: "Must not be nullptr. Capacity must be at least len elements."
            },
            {
              name: "len",
              type: "size_t",
              direction: "[in]",
              description: "Capacity of out_probabilities buffer.",
              constraints: "Must match num_classes()."
            }
          ],
          returns: {
            type: "tinyisb::Status",
            description: "tinyisb::Status::OK on success, or error status.",
            errorValues: [
              "Status::ERROR_INVALID_ARGUMENT: null pointers or length mismatch."
            ]
          },
          complexity: {
            heapAlloc: "0 bytes",
            timeComplexity: "O(C) where C is num_classes()",
            stackFootprint: "< 64 bytes"
          },
          preconditions: [
            "out_probabilities must be pre-allocated by caller."
          ],
          example: `const size_t kClasses = classifier.num_classes();
float probs[kClasses];
if (classifier.predict_proba(sensory_features, probs, kClasses) == tinyisb::Status::OK) {
    for (size_t i = 0; i < kClasses; ++i) {
        Serial.print("Class "); Serial.print(i);
        Serial.print(": "); Serial.println(probs[i]);
    }
}`
        },
        {
          id: "classifier-num-classes",
          name: "num_classes",
          signature: "size_t num_classes() const;",
          brief: "Returns total number of output classes supported by the model.",
          details: "Queries the flat dimension of the output tensor extracted during begin().",
          params: [],
          returns: {
            type: "size_t",
            description: "Total output class count (e.g. 3 for a 3-class classifier)."
          },
          complexity: {
            heapAlloc: "0 bytes",
            timeComplexity: "O(1)",
            stackFootprint: "0 bytes"
          },
          preconditions: [
            "Model should be initialized via begin()."
          ],
          example: `size_t n = classifier.num_classes();
Serial.print("Number of Classes: "); Serial.println(n);`
        },
        {
          id: "classifier-print-metadata",
          name: "print_metadata",
          signature: "void print_metadata() const;",
          brief: "Prints detailed model diagnostics to the default Serial port.",
          details: "Outputs schema version, input/output tensor shapes, quantization parameters (scale and zero-point), and total arena memory allocated versus used.",
          params: [],
          returns: {
            type: "void",
            description: "None"
          },
          complexity: {
            heapAlloc: "0 bytes",
            timeComplexity: "O(1)",
            stackFootprint: "< 64 bytes"
          },
          preconditions: [
            "Serial.begin() must have been called."
          ],
          example: `classifier.print_metadata();`
        }
      ]
    },

    Regressor: {
      name: "tinyisb::Regressor",
      category: "Estimators API",
      header: "#include <tinyISB.h>",
      brief: "High-level regressor estimator for predicting continuous scalar or vector values.",
      description: "Designed for continuous telemetry, sensor calibration, force estimation, and physiological vital sign prediction. Automatically handles INT8 input quantization and linear inverse de-quantization to produce continuous engineering units directly without manual scaling code.",
      methods: [
        {
          id: "regressor-begin",
          name: "begin",
          signature: "tinyisb::Status begin(const uint8_t* model_data, uint8_t* tensor_arena, size_t arena_size, tflite::MicroOpResolver* custom_resolver = nullptr);",
          brief: "Initialises the regressor with external model data and a static Tensor Arena.",
          details: "Validates model schema, registers operators, and maps tensors inside tensor_arena. Verifies that the model produces continuous scalar/vector outputs and extracts scale and zero-point factors.",
          params: [
            {
              name: "model_data",
              type: "const uint8_t*",
              direction: "[in]",
              description: "Pointer to the 16-byte aligned .tflite C-array stored in Flash ROM.",
              constraints: "Must not be nullptr. Must be aligned to 16 bytes (alignas(16))."
            },
            {
              name: "tensor_arena",
              type: "uint8_t*",
              direction: "[in,out]",
              description: "Pointer to contiguous pre-allocated static RAM buffer for activations.",
              constraints: "Must not be nullptr. Must be aligned to 16 bytes (alignas(16))."
            },
            {
              name: "arena_size",
              type: "size_t",
              direction: "[in]",
              description: "Total capacity of the tensor_arena buffer in bytes.",
              constraints: "Typically 4 KB to 8 KB."
            },
            {
              name: "custom_resolver",
              type: "tflite::MicroOpResolver*",
              direction: "[in]",
              description: "Optional custom op resolver.",
              constraints: "Defaults to nullptr."
            }
          ],
          returns: {
            type: "tinyisb::Status",
            description: "tinyisb::Status::OK on success, or error status code."
          },
          complexity: {
            heapAlloc: "0 bytes",
            timeComplexity: "O(K) setup overhead",
            stackFootprint: "< 256 bytes"
          },
          preconditions: [
            "Hardware clock and memory bus initialized."
          ],
          example: `constexpr size_t kArenaSize = 4 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];
static tinyisb::Regressor regressor;

regressor.begin(regressor_model_data, tensor_arena, kArenaSize);`
        },
        {
          id: "regressor-predict",
          name: "predict",
          signature: "float predict(const float* input, size_t len);",
          brief: "Predicts a continuous scalar floating-point value from an input feature vector.",
          details: "Quantizes float inputs into INT8 format, invokes forward inference, reads the output tensor, and performs automatic de-quantization: y = (q - zero_point) * scale. Returns the continuous physical value.",
          params: [
            {
              name: "input",
              type: "const float*",
              direction: "[in]",
              description: "Pointer to the input floating-point feature array.",
              constraints: "Must not be nullptr. Must contain at least 'len' floats."
            },
            {
              name: "len",
              type: "size_t",
              direction: "[in]",
              description: "Number of features in the input array.",
              constraints: "Must match input_shape().flat_size()."
            }
          ],
          returns: {
            type: "float",
            description: "Continuous scalar prediction (e.g. breaths/min, force in N, degrees).",
            errorValues: [
              "0.0f (or sets internal status) upon length mismatch or uninitialized model."
            ]
          },
          complexity: {
            heapAlloc: "0 bytes (Deterministic static execution)",
            timeComplexity: "O(M) forward pass MAC operations",
            stackFootprint: "< 128 bytes"
          },
          preconditions: [
            "regressor.begin() must have returned Status::OK."
          ],
          example: `float sensor_telemetry[6] = {-1.23f, 0.11f, 1.33f, 0.31f, -0.61f, 0.46f};
float estimated_val = regressor.predict(sensor_telemetry, 6);
Serial.print("Estimated Metric: ");
Serial.println(estimated_val, 3);`
        },
        {
          id: "regressor-predict-vector",
          name: "predict_vector",
          signature: "tinyisb::Status predict_vector(const float* input, size_t in_len, float* output, size_t out_len);",
          brief: "Predicts a multi-dimensional continuous output vector.",
          details: "Supports multi-target regression (e.g. 3D spatial velocity [vx, vy, vz] or multi-joint angles). De-quantizes each element of the output tensor into the caller's output buffer.",
          params: [
            {
              name: "input",
              type: "const float*",
              direction: "[in]",
              description: "Pointer to input feature array."
            },
            {
              name: "in_len",
              type: "size_t",
              direction: "[in]",
              description: "Number of input elements."
            },
            {
              name: "output",
              type: "float*",
              direction: "[out]",
              description: "Destination array for predicted continuous outputs."
            },
            {
              name: "out_len",
              type: "size_t",
              direction: "[in]",
              description: "Capacity of output buffer (must match output_dimension())."
            }
          ],
          returns: {
            type: "tinyisb::Status",
            description: "Status::OK upon success."
          },
          complexity: {
            heapAlloc: "0 bytes",
            timeComplexity: "O(M) forward inference",
            stackFootprint: "< 128 bytes"
          },
          preconditions: [
            "out_len == output_dimension()."
          ],
          example: `float out_vector[3];
if (regressor.predict_vector(inputs, 6, out_vector, 3) == tinyisb::Status::OK) {
    Serial.println("Multi-target prediction succeeded.");
}`
        },
        {
          id: "regressor-output-dimension",
          name: "output_dimension",
          signature: "size_t output_dimension() const;",
          brief: "Returns number of continuous scalar outputs produced by the model.",
          details: "Returns 1 for single-target regressors or N for multi-target regression networks.",
          params: [],
          returns: {
            type: "size_t",
            description: "Output dimension count."
          },
          complexity: {
            heapAlloc: "0 bytes",
            timeComplexity: "O(1)",
            stackFootprint: "0 bytes"
          },
          preconditions: [
            "Model initialized."
          ],
          example: `size_t dim = regressor.output_dimension();`
        }
      ]
    },

    Engine: {
      name: "tinyisb::Engine",
      category: "Core Engine",
      header: "#include <core/Engine.h>",
      brief: "Core TensorFlow Lite Micro wrapper managing tensor buffers and execution.",
      description: "Low-level inference engine responsible for FlatBuffer schema verification, operator dispatching, static tensor arena allocation via tflite::MicroInterpreter, zero-copy quantization scaling, and memory diagnostics.",
      methods: [
        {
          id: "engine-begin",
          name: "begin",
          signature: "Status begin(const uint8_t* model_data, uint8_t* tensor_arena, size_t arena_size, tflite::MicroOpResolver* custom_resolver = nullptr);",
          brief: "Binds model data and allocates tensor arena buffers.",
          details: "Instantiates MicroInterpreter, executes AllocateTensors(), verifies arena capacity, and inspects tensor descriptors.",
          params: [
            { name: "model_data", type: "const uint8_t*", direction: "[in]", description: "Flash model array pointer." },
            { name: "tensor_arena", type: "uint8_t*", direction: "[in,out]", description: "Static RAM activation buffer." },
            { name: "arena_size", type: "size_t", direction: "[in]", description: "Buffer byte capacity." },
            { name: "custom_resolver", type: "tflite::MicroOpResolver*", direction: "[in]", description: "Custom op resolver pointer." }
          ],
          returns: { type: "Status", description: "Status::OK on success." },
          complexity: { heapAlloc: "0 bytes", timeComplexity: "O(K)", stackFootprint: "< 256 bytes" },
          preconditions: ["16-byte alignment of model_data and tensor_arena."],
          example: `tinyisb::Engine engine;
engine.begin(model_data, arena, 4096);`
        },
        {
          id: "engine-invoke",
          name: "invoke",
          signature: "Status invoke();",
          brief: "Executes forward model inference.",
          details: "Calls tflite::MicroInterpreter::Invoke(). Guarantees no dynamic heap allocations are made during execution.",
          params: [],
          returns: { type: "Status", description: "Status::OK or Status::ERROR_INVOKE_FAILED." },
          complexity: { heapAlloc: "0 bytes", timeComplexity: "O(M) network MAC ops", stackFootprint: "< 128 bytes" },
          preconditions: ["Input tensors must be populated prior to calling invoke()."],
          example: `if (engine.invoke() == tinyisb::Status::OK) { /* Inference succeeded */ }`
        },
        {
          id: "engine-input-shape",
          name: "input_shape",
          signature: "const TensorShape& input_shape() const;",
          brief: "Retrieves shape dimensions of the model's input tensor.",
          details: "Returns TensorShape containing dimensions (e.g. [1, 128, 1]) and rank.",
          params: [],
          returns: { type: "const TensorShape&", description: "Reference to cached input shape descriptor." },
          complexity: { heapAlloc: "0 bytes", timeComplexity: "O(1)", stackFootprint: "0 bytes" },
          preconditions: ["Engine initialized."],
          example: `const tinyisb::TensorShape& shape = engine.input_shape();
Serial.print("Input flat size: "); Serial.println(shape.flat_size());`
        },
        {
          id: "engine-output-shape",
          name: "output_shape",
          signature: "const TensorShape& output_shape() const;",
          brief: "Retrieves shape dimensions of the model's output tensor.",
          details: "Returns TensorShape containing output dimensions (e.g. [1, 3]).",
          params: [],
          returns: { type: "const TensorShape&", description: "Reference to cached output shape descriptor." },
          complexity: { heapAlloc: "0 bytes", timeComplexity: "O(1)", stackFootprint: "0 bytes" },
          preconditions: ["Engine initialized."],
          example: `const tinyisb::TensorShape& shape = engine.output_shape();`
        },
        {
          id: "engine-arena-used-bytes",
          name: "arena_used_bytes",
          signature: "size_t arena_used_bytes() const;",
          brief: "Returns actual number of bytes consumed in the Tensor Arena.",
          details: "Queries tflite::MicroInterpreter::arena_used_bytes() to allow precise RAM tuning.",
          params: [],
          returns: { type: "size_t", description: "Allocated activation bytes in RAM." },
          complexity: { heapAlloc: "0 bytes", timeComplexity: "O(1)", stackFootprint: "0 bytes" },
          preconditions: ["Engine initialized."],
          example: `Serial.print("RAM Used: ");
Serial.print(engine.arena_used_bytes());
Serial.println(" bytes");`
        }
      ]
    },

    Status: {
      name: "tinyisb::Status",
      category: "Types & Enums",
      header: "#include <core/Types.h>",
      brief: "Strongly-typed enumeration for return codes and error diagnostics.",
      description: "tinyISB avoids C++ exceptions to minimize Flash footprint and ensure deterministic hard real-time behavior. All operational methods return a Status enum code.",
      values: [
        { name: "Status::OK", value: 0, description: "Operation completed successfully without errors." },
        { name: "Status::ERROR_INVALID_ARGUMENT", value: 1, description: "Null pointer provided, buffer capacity zero, or invalid arguments." },
        { name: "Status::ERROR_SCHEMA_MISMATCH", value: 2, description: "FlatBuffers schema version mismatch between model array and TFLM runtime." },
        { name: "Status::ERROR_ALLOCATION_FAILED", value: 3, description: "Provided tensor_arena RAM buffer is too small to fit model activation tensors." },
        { name: "Status::ERROR_INVOKE_FAILED", value: 4, description: "Execution error during tflite::MicroInterpreter::Invoke()." },
        { name: "Status::ERROR_NOT_INITIALIZED", value: 5, description: "Attempted inference or metadata query before calling begin()." },
        { name: "Status::ERROR_DIMENSION_MISMATCH", value: 6, description: "Input or output array length does not match model tensor shape." },
        { name: "Status::ERROR_UNSUPPORTED_TYPE", value: 7, description: "Tensor data type is not supported (only Float32, INT8, and UInt8 supported)." }
      ]
    },

    TensorShape: {
      name: "tinyisb::TensorShape",
      category: "Types & Enums",
      header: "#include <core/Types.h>",
      brief: "Encapsulates up to 4-dimensional tensor shape metadata.",
      description: "Stores rank (ndim) and dimension sizes (dims[4]). Provides flat_size() to compute the total element count.",
      properties: [
        { name: "dims[4]", type: "int32_t", description: "Array storing dimension sizes [batch, dim1, dim2, dim3]." },
        { name: "ndim", type: "size_t", description: "Rank or number of active dimensions (1 to 4)." }
      ],
      methods: [
        {
          id: "tensorshape-flat-size",
          name: "flat_size",
          signature: "size_t flat_size() const;",
          brief: "Calculates total number of elements across all dimensions.",
          details: "Computes the product of all active dimensions: dims[0] * dims[1] * ... * dims[ndim-1].",
          params: [],
          returns: { type: "size_t", description: "Product of dimension sizes." },
          complexity: { heapAlloc: "0 bytes", timeComplexity: "O(1)", stackFootprint: "0 bytes" },
          example: `size_t total_elements = shape.flat_size();`
        }
      ]
    },

    QuantParams: {
      name: "tinyisb::QuantParams",
      category: "Types & Enums",
      header: "#include <core/Types.h>",
      brief: "Stores quantization scale and zero-point parameters.",
      description: "Defines affine quantization mapping: real_value = (quantized_value - zero_point) * scale.",
      properties: [
        { name: "scale", type: "float", description: "Quantization scale factor S." },
        { name: "zero_point", type: "int32_t", description: "Quantization zero-point offset Z." },
        { name: "is_quantized", type: "bool", description: "True if tensor is quantized (INT8/UInt8); false if Float32." }
      ]
    },

    ModelMetadata: {
      name: "tinyisb::ModelMetadata",
      category: "Types & Enums",
      header: "#include <core/Types.h>",
      brief: "Comprehensive model metadata inspection structure.",
      description: "Aggregates tensor shapes, quantization parameters, and memory arena utilization metrics.",
      properties: [
        { name: "schema_version", type: "uint32_t", description: "TFLite schema version." },
        { name: "arena_size_bytes", type: "size_t", description: "Total capacity of the provided Tensor Arena." },
        { name: "arena_used_bytes", type: "size_t", description: "Actual bytes used by activation tensors." },
        { name: "input_shape", type: "TensorShape", description: "Input tensor dimensions." },
        { name: "output_shape", type: "TensorShape", description: "Output tensor dimensions." },
        { name: "input_quant", type: "QuantParams", description: "Input quantization parameters." },
        { name: "output_quant", type: "QuantParams", description: "Output quantization parameters." }
      ]
    }
  },

  examples: [
    {
      id: "BasicClassification",
      title: "Basic Sensory Classification",
      model: "Dense MLP (4 -> 16 -> 8 -> 3)",
      inputShape: "[1, 4]",
      quantization: "INT8",
      flashSize: "2,864 bytes",
      arenaRAM: "4 KB",
      latencyRP2040: "~0.42 ms",
      description: "Multi-class tabular classifier for sensor fusion, vibration state diagnostics, and ambient condition tracking.",
      link: "examples/BasicClassification/BasicClassification.ino"
    },
    {
      id: "ECGArrhythmiaClassifier",
      title: "1D-CNN ECG Arrhythmia Classifier",
      model: "Conv1D + MaxPool + Conv1D + GAP + Dense",
      inputShape: "[1, 128, 1]",
      quantization: "INT8",
      flashSize: "5,616 bytes",
      arenaRAM: "10 KB",
      latencyRP2040: "~3.18 ms",
      description: "Clinical biosignal processing classifying 250 Hz single-lead ECG into Normal Sinus Rhythm (NSR), Tachycardia, and Bradycardia.",
      link: "examples/ECGArrhythmiaClassifier/ECGArrhythmiaClassifier.ino"
    },
    {
      id: "EMGGestureClassifier",
      title: "Distilled sEMG Hand Gesture Classifier",
      model: "Student MLP (64 -> 32 -> 16 -> 4)",
      inputShape: "[1, 64]",
      quantization: "INT8",
      flashSize: "3,968 bytes",
      arenaRAM: "6 KB",
      latencyRP2040: "~0.85 ms",
      description: "Knowledge Distillation transferring temporal features from a heavy BiLSTM Teacher to a lightweight edge Student for myoelectric prosthetics.",
      link: "examples/EMGGestureClassifier/EMGGestureClassifier.ino"
    },
    {
      id: "SignalRegressor",
      title: "Continuous Signal Parameter Regressor",
      model: "Regression MLP (6 -> 20 -> 12 -> 1)",
      inputShape: "[1, 6]",
      quantization: "INT8",
      flashSize: "2,528 bytes",
      arenaRAM: "4 KB",
      latencyRP2040: "~0.38 ms",
      description: "Continuous physiological parameter regression estimating respiration rate in Breaths Per Minute directly from multi-channel telemetry.",
      link: "examples/SignalRegressor/SignalRegressor.ino"
    }
  ],

  hardwareTargets: [
    {
      board: "Raspberry Pi Pico",
      arch: "ARM Cortex-M0+ (Dual Core @ 133 MHz)",
      core: "rp2040:rp2040",
      fqbn: "rp2040:rp2040:rpipico",
      macros: "ARDUINO_ARCH_RP2040, PICO_RP2040",
      pioBoard: "pico",
      resolver: "tflite::AllOpsResolver"
    },
    {
      board: "Raspberry Pi Pico 2",
      arch: "ARM Cortex-M33 with DSP & FPU (Dual Core @ 150 MHz)",
      core: "rp2040:rp2040",
      fqbn: "rp2040:rp2040:rpipico2",
      macros: "ARDUINO_ARCH_RP2040, PICO_RP2350",
      pioBoard: "pico",
      resolver: "tflite::AllOpsResolver"
    },
    {
      board: "Espressif ESP32-S3",
      arch: "Xtensa LX7 with Vector SIMD (Dual Core @ 240 MHz)",
      core: "esp32:esp32",
      fqbn: "esp32:esp32:esp32s3",
      macros: "ESP32, CONFIG_IDF_TARGET_ESP32S3",
      pioBoard: "esp32-s3-devkitc-1",
      resolver: "tinyisb::internal::UniversalOpResolver"
    },
    {
      board: "Espressif ESP32-C6",
      arch: "32-bit RISC-V RV32IMAC (@ 160 MHz)",
      core: "esp32:esp32",
      fqbn: "esp32:esp32:esp32c6",
      macros: "ESP32, CONFIG_IDF_TARGET_ESP32C6",
      pioBoard: "esp32-c6-devkitc-1",
      resolver: "tinyisb::internal::UniversalOpResolver"
    },
    {
      board: "Arduino Nano 33 BLE Sense",
      arch: "ARM Cortex-M4F with FPU (@ 64 MHz)",
      core: "arduino:mbed_nano",
      fqbn: "arduino:mbed_nano:nano33ble",
      macros: "ARDUINO_ARDUINO_NANO33BLE, TARGET_NRF52840",
      pioBoard: "nano33ble",
      resolver: "tflite::AllOpsResolver"
    }
  ]
};
