#ifndef GOONNX_ORT_TENSOR_TYPE_AND_SHAPE_INFO
#define GOONNX_ORT_TENSOR_TYPE_AND_SHAPE_INFO

#include <onnxruntime_c_api.h>

typedef struct OrtGetTensorElementTypeResponse {
	ONNXTensorElementDataType dataType;
	OrtStatus *status;
} OrtGetTensorElementTypeResponse;

typedef struct OrtGetDimensionsCountResponse {
	size_t numDims;
	OrtStatus *status;
} OrtGetDimensionsCountResponse;

OrtGetTensorElementTypeResponse getTensorElementType(OrtApi *api, OrtTensorTypeAndShapeInfo *typeInfo);
OrtGetDimensionsCountResponse getDimensionsCount(OrtApi *api, OrtTensorTypeAndShapeInfo *typeInfo);
OrtStatus* getDimensions(OrtApi *api, OrtTensorTypeAndShapeInfo *typeInfo, size_t numDims, int64_t *resultContainer);

#endif