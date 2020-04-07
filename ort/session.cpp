extern "C" {
    #include <onnxruntime_c_api.h>
    #include "session.h"

    OrtCreateSessionResponse createSession(OrtApi *api, OrtEnv *env, const char *modelPath, OrtSessionOptions *sessionOptions) {
        OrtSession *session;
        OrtStatus *status;

        status = api->CreateSession(env, modelPath, sessionOptions, &session);

        OrtCreateSessionResponse response;
        response.session = session;
        response.status = status;

        return response;
    }

    void releaseSession(OrtApi *api, OrtSession *session) {
        api->ReleaseSession(session);
    }

    OrtGetIOCountResponse getInputCount(OrtApi *api, OrtSession *session) {
        size_t numInputNodes;
        OrtStatus *status;

        status = api->SessionGetInputCount(session, &numInputNodes);

        OrtGetIOCountResponse response;
        response.numNodes = numInputNodes;
        response.status = status;

        return response;
    }

    OrtGetIONameResponse getInputName(OrtApi *api, OrtSession *session, size_t i, OrtAllocator *allocator) {
        char *inputName;
        OrtStatus *status;

        api->SessionGetInputName(session, i, allocator, &inputName);

        OrtGetIONameResponse response;
        response.name = inputName;
        response.status = status;

        return response;
    }

    OrtGetIOTypeInfoResponse getInputTypeInfo(OrtApi *api, OrtSession *session, size_t i) {
        OrtTypeInfo *typeInfo;
        OrtStatus *status;

        status = api->SessionGetInputTypeInfo(session, i, &typeInfo);

        OrtGetIOTypeInfoResponse response;
        response.typeInfo = typeInfo;
        response.status = status;

        return response;
    }
}