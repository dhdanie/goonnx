extern "C" {
    #include <onnxruntime_c_api.h>
    #include "memory-info.h"

    OrtCreateCpuMemoryInfoResponse createCpuMemoryInfo(OrtApi *api, OrtAllocatorType allocatorType, OrtMemType memType) {
        OrtMemoryInfo *memoryInfo;
        OrtStatus *status;

        status = api->CreateCpuMemoryInfo(allocatorType, memType, &memoryInfo);

        OrtCreateCpuMemoryInfoResponse response;
        response.memoryInfo = memoryInfo;
        response.status = status;

        return response;
    }

    void releaseMemoryInfo(OrtApi *api, OrtMemoryInfo *memoryInfo) {
        api->ReleaseMemoryInfo(memoryInfo);
    }
}