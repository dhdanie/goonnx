extern "C" {
    #include <onnxruntime_c_api.h>
    #include "tensor-type-and-shape-info.h"

    OrtGetTensorElementTypeResponse getTensorElementType(OrtApi *api, OrtTensorTypeAndShapeInfo *typeInfo) {
        ONNXTensorElementDataType dataType;
        OrtStatus *status;

        status = api->GetTensorElementType(typeInfo, &dataType);

        OrtGetTensorElementTypeResponse response;
        response.dataType = dataType;
        response.status = status;

        return response;
    }

    OrtGetDimensionsCountResponse getDimensionsCount(OrtApi *api, OrtTensorTypeAndShapeInfo *typeInfo) {
        size_t numDims;
        OrtStatus *status;

        status = api->GetDimensionsCount(typeInfo, &numDims);

        OrtGetDimensionsCountResponse response;
        response.numDims = numDims;
        response.status = status;

        return response;
    }

    OrtStatus* getDimensions(OrtApi *api, OrtTensorTypeAndShapeInfo *typeInfo, size_t numDims, int64_t *resultContainer) {
        return api->GetDimensions(typeInfo, resultContainer, numDims);
    }
}