#ifndef GOONNX_ORT_SESSION
#define GOONNX_ORT_SESSION

#include <onnxruntime_c_api.h>

typedef struct OrtCreateSessionResponse {
	OrtSession *session;
	OrtStatus *status;
} OrtCreateSessionResponse;

typedef struct OrtGetIOCountResponse {
	size_t numNodes;
	OrtStatus *status;
} OrtGetIOCountResponse;

typedef struct OrtGetIONameResponse {
	char *name;
	OrtStatus *status;
} OrtGetIONameResponse;

typedef struct OrtGetIOTypeInfoResponse {
	OrtTypeInfo *typeInfo;
	OrtStatus *status;
} OrtGetIOTypeInfoResponse;

typedef struct OrtRunResponse {
    OrtValue *output;
    OrtStatus *status;
} OrtRunResponse;

OrtCreateSessionResponse createSession(OrtApi *api, OrtEnv *env, const char *modelPath,
        OrtSessionOptions *sessionOptions);
void releaseSession(OrtApi *api, OrtSession *session);
OrtGetIOCountResponse getInputCount(OrtApi *api, OrtSession *session);
OrtGetIONameResponse getInputName(OrtApi *api, OrtSession *session, size_t i, OrtAllocator *allocator);
OrtGetIOTypeInfoResponse getInputTypeInfo(OrtApi *api, OrtSession *session, size_t i);
OrtGetIOCountResponse getOutputCount(OrtApi *api, OrtSession *session);
OrtGetIONameResponse getOutputName(OrtApi *api, OrtSession *session, size_t i, OrtAllocator *allocator);
OrtGetIOTypeInfoResponse getOutputTypeInfo(OrtApi *api, OrtSession *session, size_t i);
OrtRunResponse run(OrtApi *api, OrtSession *session, OrtRunOptions *runOptions, char **inputNames, OrtValue **input,
        size_t inputLen, char **outputNames, size_t outputNamesLen);

#endif