#ifndef GOONNX_ORT_ENVIRONMENT
#define GOONNX_ORT_ENVIRONMENT

#include <onnxruntime_c_api.h>

typedef struct OrtCreateEnvResponse {
    const OrtEnv *env;
    OrtStatus *status;
} OrtCreateEnvResponse;

OrtCreateEnvResponse createEnv(OrtApi* api, OrtLoggingLevel level, char* logId);
OrtCreateEnvResponse createEnvWithCustomLogger(OrtApi *api, void *params, OrtLoggingLevel level, char *logId);
void releaseEnv(OrtApi* api, OrtEnv* env);
extern void logCustom(void *param, OrtLoggingLevel severity, char *category, char *logId, char *codeLocation, char *message);

#endif