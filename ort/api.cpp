extern "C" {
    #include <onnxruntime_c_api.h>
    #include <string.h>
    #include "api.h"

    const OrtApi* getApi() {
        return OrtGetApiBase()->GetApi(ORT_API_VERSION);
    }

    const char* parseStatus(OrtApi* api, OrtStatus* status) {
        if(status != NULL) {
            const char* msg = api->GetErrorMessage(status);
            char *copy = (char *)malloc(strlen(msg));
            strcpy(copy, msg);
            api->ReleaseStatus(status);
            return copy;
        }
        return NULL;
    }
}