#ifndef GOONNX_ORT_MEMORY_INFO
#define GOONNX_ORT_MEMORY_INFO

#include <onnxruntime_c_api.h>

typedef struct OrtCreateCpuMemoryInfoResponse {
    OrtMemoryInfo *memoryInfo;
    OrtStatus *status;
} OrtCreateCpuMemoryInfoResponse;

OrtCreateCpuMemoryInfoResponse createCpuMemoryInfo(OrtApi *api, OrtAllocatorType allocatorType, OrtMemType memType);
void releaseMemoryInfo(OrtApi *api, OrtMemoryInfo *memoryInfo);

#endif