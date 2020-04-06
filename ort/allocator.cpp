extern "C" {
    #include <onnxruntime_c_api.h>
    #include "allocator.h"

    GetAllocatorResponse getAllocatorWithDefaultOptions(OrtApi *api) {
        OrtAllocator *allocator;
        OrtStatus *status;

        status = api->GetAllocatorWithDefaultOptions(&allocator);

        GetAllocatorResponse response;
        response.allocator = allocator;
        response.status = status;

        return response;
    }
}