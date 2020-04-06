#ifndef GOONNX_ORT_SESSION_OPTIONS
#define GOONNX_ORT_SESSION_OPTIONS

#include <onnxruntime_c_api.h>

typedef struct OrtCreateSessionOptionsResponse {
	OrtSessionOptions *sessionOptions;
	OrtStatus *status;
} OrtCreateSessionOptionsResponse;

OrtCreateSessionOptionsResponse createSessionOptions(OrtApi *api);
void releaseSessionOptions(OrtApi *api, OrtSessionOptions *sessionOptions);
OrtStatus* setIntraOpNumThreads(OrtApi *api, OrtSessionOptions *sessionOptions, int numThreads);
OrtStatus* setSessionGraphOptimizationLevel(OrtApi *api, OrtSessionOptions *sessionOptions, GraphOptimizationLevel level);

#endif