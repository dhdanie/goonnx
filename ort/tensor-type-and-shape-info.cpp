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

    OrtGetDimensionsResponse getDimensions(OrtApi *api, OrtTensorTypeAndShapeInfo *typeInfo, size_t numDims) {
        int64_t *dims;
        OrtStatus *status;

        dims = (int64_t *)malloc(numDims * sizeof(int64_t));

        status = api->GetDimensions(typeInfo, dims, numDims);

        OrtGetDimensionsResponse response;
        response.dims = dims;
        response.status = status;

        return response;
    }

    OrtGetSymbolicDimensionsResponse getSymbolicDimensions(OrtApi *api, OrtTensorTypeAndShapeInfo *typeInfo, size_t numDims) {
        const char *dimParams;
        OrtStatus *status;

        status = api->GetSymbolicDimensions(typeInfo, &dimParams, numDims);

        OrtGetSymbolicDimensionsResponse response;
        response.status = status;
        response.dimParams = dimParams;

        return response;
    }


}