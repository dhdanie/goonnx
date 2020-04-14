extern "C" {
    #include <onnxruntime_c_api.h>
    #include "environment.h"

    void logCustomWrapper(void *params, OrtLoggingLevel severity, const char *category, const char *logId, const char *codeLocation, const char *message) {
    	logCustom(params, severity, (char *)category, (char *)logId, (char *)codeLocation, (char *) message);
    }

    OrtCreateEnvResponse createEnv(OrtApi* api, OrtLoggingLevel level, char* logId) {
        OrtEnv* env;
        OrtStatus* status;

        status = api->CreateEnv(level, logId, &env);

        OrtCreateEnvResponse response;
        response.env = env;
        response.status = status;

        return response;
    }

    OrtCreateEnvResponse createEnvWithCustomLogger(OrtApi* api, void *params, OrtLoggingLevel level, char* logId) {
        OrtEnv* env;
        OrtStatus* status;

        status = api->CreateEnvWithCustomLogger(logCustomWrapper, params, level, logId, &env);

        OrtCreateEnvResponse response;
        response.env = env;
        response.status = status;

        return response;
    }

    void releaseEnv(OrtApi* api, OrtEnv* env) {
        api->ReleaseEnv(env);
    }
}