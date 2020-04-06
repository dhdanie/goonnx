#ifndef GOONNX_ORT_SESSION
#define GOONNX_ORT_SESSION

#include <onnxruntime_c_api.h>

typedef struct OrtCreateSessionResponse {
	OrtSession *session;
	OrtStatus *status;
} OrtCreateSessionResponse;

typedef struct OrtGetInputCountResponse {
	size_t numInputNodes;
	OrtStatus *status;
} OrtGetInputCountResponse;

typedef struct OrtGetInputNameResponse {
	char *inputName;
	OrtStatus *status;
} OrtGetInputNameResponse;

typedef struct OrtGetInputTypeInfoResponse {
	OrtTypeInfo *typeInfo;
	OrtStatus *status;
} OrtGetInputTypeInfoResponse;

OrtCreateSessionResponse createSession(OrtApi *api, OrtEnv *env, const char *modelPath, OrtSessionOptions *sessionOptions);
void releaseSession(OrtApi *api, OrtSession *session);
OrtGetInputCountResponse getInputCount(OrtApi *api, OrtSession *session);
OrtGetInputNameResponse getInputName(OrtApi *api, OrtSession *session, size_t i, OrtAllocator *allocator);
OrtGetInputTypeInfoResponse getInputTypeInfo(OrtApi *api, OrtSession *session, size_t i);

#endif