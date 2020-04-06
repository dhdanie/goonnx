extern "C" {
    #include <onnxruntime_c_api.h>
    #include "api.h"

    const OrtApi* getApi() {
        return OrtGetApiBase()->GetApi(ORT_API_VERSION);
    }

    const char* parseStatus(OrtApi* api, OrtStatus* status) {
        if(status != NULL) {
            const char* msg = api->GetErrorMessage(status);
            api->ReleaseStatus(status);
            return msg;
        }
        return NULL;
    }
}