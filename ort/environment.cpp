extern "C" {
    #include <onnxruntime_c_api.h>
    #include "environment.h"

    OrtCreateEnvResponse createEnv(OrtApi* api, OrtLoggingLevel level, char* logId) {
        OrtEnv* env;
        OrtStatus* status;

        status = api->CreateEnv(level, logId, &env);

        OrtCreateEnvResponse response;
        response.env = env;
        response.status = status;

        return response;
    }

    void releaseEnv(OrtApi* api, OrtEnv* env) {
        api->ReleaseEnv(env);
    }
}