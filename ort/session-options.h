#ifndef GOONNX_ORT_SESSION_OPTIONS
#define GOONNX_ORT_SESSION_OPTIONS

#include <onnxruntime_c_api.h>

typedef struct OrtCreateSessionOptionsParams {
	ORTCHAR_T *optimizedModelFilePath;
	ExecutionMode executionMode;
	int profilingEnabled;
	const ORTCHAR_T *profileFilePrefix;
	int memPatternEnabled;
	int cpuMemArenaEnabled;
	const char *logId;
	int logVerbosityLevel;
	int logSeverityLevel;
	GraphOptimizationLevel graphOptimizationLevel;
	int intraOpNumThreads;
	int interOpNumThreads;
	int numCustomOpDomains;
	OrtCustomOpDomain **customOpDomains;
} OrtCreateSessionOptionsParams;

typedef struct OrtCreateSessionOptionsResponse {
	OrtSessionOptions *sessionOptions;
	OrtStatus *status;
} OrtCreateSessionOptionsResponse;

#define DefaultExecutionMode ORT_SEQUENTIAL
#define DefaultGraphOptimizationLevel ORT_ENABLE_ALL
#define DefaultIntraOpNumThreads 0
#define DefaultInterOpNumThreads 0

OrtCreateSessionOptionsResponse createSessionOptions(OrtApi *api, OrtCreateSessionOptionsParams *params);
OrtCreateSessionOptionsResponse releaseAndRespondErrorStatus(OrtApi *api, OrtSessionOptions *sessionOptions, OrtStatus *status);
OrtCreateSessionOptionsResponse respondErrorStatus(OrtStatus *status);

#endif