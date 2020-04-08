extern "C" {
    #include <onnxruntime_c_api.h>
    #include "value.h"

    OrtCreateTensorWithDataAsOrtValueResponse createTensorWithDataAsOrtValue(OrtApi *api, OrtMemoryInfo *memoryInfo,
            void *data, size_t dataLen, int64_t *shape, size_t shapeLen, ONNXTensorElementDataType type) {
        OrtValue *value;
        OrtStatus *status;

        status = api->CreateTensorWithDataAsOrtValue(memoryInfo, data, dataLen, shape, shapeLen, type, &value);

        OrtCreateTensorWithDataAsOrtValueResponse response;
        response.value = value;
        response.status = status;

        return response;
    }

    OrtIsTensorResponse isTensor(OrtApi *api, OrtValue *value) {
        int isTensor;
        OrtStatus *status;

        status = api->IsTensor(value, &isTensor);

        OrtIsTensorResponse response;
        response.isTensor = isTensor;
        response.status = status;

        return response;
    }

    OrtGetTensorMutableFloatDataResponse getTensorMutableFloatData(OrtApi *api, OrtValue *value) {
        float *out;
        OrtStatus *status;

        status = api->GetTensorMutableData(value, (void **)&out);

        OrtGetTensorMutableFloatDataResponse response;
        response.status = status;
        response.out = out;

        return response;
    }
}