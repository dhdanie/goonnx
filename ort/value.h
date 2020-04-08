#ifndef GOONNX_ORT_VALUE
#define GOONNX_ORT_VALUE

#include <onnxruntime_c_api.h>

typedef struct OrtCreateTensorWithDataAsOrtValueResponse {
    OrtValue *value;
    OrtStatus *status;
} OrtCreateTensorWithDataAsOrtValueResponse;

typedef struct OrtIsTensorResponse {
    int isTensor;
    OrtStatus *status;
} OrtIsTensorResponse;

typedef struct OrtGetTensorMutableFloatDataResponse {
    float *out;
    OrtStatus *status;
} OrtGetTensorMutableFloatDataResponse;

OrtCreateTensorWithDataAsOrtValueResponse createTensorWithDataAsOrtValue(OrtApi *api, OrtMemoryInfo *memoryInfo,
        void *data, size_t dataLen, int64_t *shape, size_t shapeLen, ONNXTensorElementDataType type);

OrtIsTensorResponse isTensor(OrtApi *api, OrtValue *value);

OrtGetTensorMutableFloatDataResponse getTensorMutableFloatData(OrtApi *api, OrtValue *value);

#endif