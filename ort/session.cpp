extern "C" {
    #include <onnxruntime_c_api.h>
    #include <stdio.h>
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

    void releaseSessionOptions(OrtApi *api, OrtSessionOptions *opts) {
        api->ReleaseSessionOptions(opts);
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

    OrtGetIOCountResponse getOutputCount(OrtApi *api, OrtSession *session) {
        size_t numOutputNodes;
        OrtStatus *status;

        status = api->SessionGetOutputCount(session, &numOutputNodes);

        OrtGetIOCountResponse response;
        response.numNodes = numOutputNodes;
        response.status = status;

        return response;
    }

    OrtGetIONameResponse getOutputName(OrtApi *api, OrtSession *session, size_t i, OrtAllocator *allocator) {
        char *outputName;
        OrtStatus *status;

        api->SessionGetOutputName(session, i, allocator, &outputName);

        OrtGetIONameResponse response;
        response.name = outputName;
        response.status = status;

        return response;
    }

    OrtGetIOTypeInfoResponse getOutputTypeInfo(OrtApi *api, OrtSession *session, size_t i) {
        OrtTypeInfo *typeInfo;
        OrtStatus *status;

        status = api->SessionGetOutputTypeInfo(session, i, &typeInfo);

        OrtGetIOTypeInfoResponse response;
        response.typeInfo = typeInfo;
        response.status = status;

        return response;
    }

    OrtRunResponse run(OrtApi *api, OrtSession *session, OrtRunOptions *runOptions, char **inputNames, OrtValue **input,
            size_t inputLen, char **outputNames, size_t outputNamesLen) {
        OrtValue *output = NULL;
        OrtStatus *status;

        status = api->Run(session, runOptions, inputNames, input, inputLen, outputNames, outputNamesLen, &output);

        OrtRunResponse response;
        response.output = output;
        response.status = status;

        return response;
    }
}