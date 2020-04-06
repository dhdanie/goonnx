extern "C" {
    #include <onnxruntime_c_api.h>
    #include "session-options.h"

    OrtCreateSessionOptionsResponse createSessionOptions(OrtApi *api) {
        OrtSessionOptions *sessionOptions;
        OrtStatus *status;

        status = api->CreateSessionOptions(&sessionOptions);

        OrtCreateSessionOptionsResponse response;
        response.sessionOptions = sessionOptions;
        response.status = status;

        return response;
    }

    void releaseSessionOptions(OrtApi *api, OrtSessionOptions *sessionOptions) {
        api->ReleaseSessionOptions(sessionOptions);
    }

    OrtStatus* setIntraOpNumThreads(OrtApi *api, OrtSessionOptions *sessionOptions, int numThreads) {
        return api->SetIntraOpNumThreads(sessionOptions, numThreads);
    }

    OrtStatus* setSessionGraphOptimizationLevel(OrtApi *api, OrtSessionOptions *sessionOptions, GraphOptimizationLevel level) {
        return api->SetSessionGraphOptimizationLevel(sessionOptions, level);
    }
}