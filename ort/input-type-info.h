#ifndef GOONNX_ORT_INPUT_TYPE_INFO
#define GOONNX_ORT_INPUT_TYPE_INFO

#include <onnxruntime_c_api.h>

typedef struct OrtCastTypeInfoToTensorInfoResponse {
	const OrtTensorTypeAndShapeInfo *tensorInfo;
	OrtStatus *status;
} OrtCastTypeInfoToTensorInfoResponse;

void releaseTypeInfo(OrtApi *api, OrtTypeInfo *typeInfo);
OrtCastTypeInfoToTensorInfoResponse castTypeInfoToTensorInfo(OrtApi *api, OrtTypeInfo *typeInfo);

#endif