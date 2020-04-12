extern "C" {
    #include <onnxruntime_c_api.h>
    #include "session-options.h"

	OrtCreateSessionOptionsResponse createSessionOptions(OrtApi *api, OrtCreateSessionOptionsParams *params) {
        OrtStatus *status;
        OrtSessionOptions *sessionOptions;

        status = api->CreateSessionOptions(&sessionOptions);
        if(status != NULL) {
            return respondErrorStatus(status);
        }

        if(params->optimizedModelFilePath != NULL) {
            status = api->SetOptimizedModelFilePath(sessionOptions, params->optimizedModelFilePath);
            if(status != NULL) {
                return releaseAndRespondErrorStatus(api, sessionOptions, status);
            }
        }
        if(params->executionMode != 0) {
            status = api->SetSessionExecutionMode(sessionOptions, params->executionMode);
            if(status != NULL) {
                return releaseAndRespondErrorStatus(api, sessionOptions, status);
            }
        }
        if(params->profilingEnabled == 1 && params->profileFilePrefix != NULL) {
            status = api->EnableProfiling(sessionOptions, params->profileFilePrefix);
            if(status != NULL) {
                return releaseAndRespondErrorStatus(api, sessionOptions, status);
            }
        }
        if(params->memPatternEnabled == 1) {
            status = api->EnableMemPattern(sessionOptions);
            if(status != NULL) {
                return releaseAndRespondErrorStatus(api, sessionOptions, status);
            }
        }
        if(params->cpuMemArenaEnabled == 1) {
            status = api->EnableCpuMemArena(sessionOptions);
            if(status != NULL) {
                return releaseAndRespondErrorStatus(api, sessionOptions, status);
            }
        }
        if(params->logId != NULL) {
            status = api->SetSessionLogId(sessionOptions, params->logId);
            if(status != NULL) {
                return releaseAndRespondErrorStatus(api, sessionOptions, status);
            }
        }
        if(params->logVerbosityLevel > 0) {
            status = api->SetSessionLogVerbosityLevel(sessionOptions, params->logVerbosityLevel);
            if(status != NULL) {
                return releaseAndRespondErrorStatus(api, sessionOptions, status);
            }
        }
        if(params->logSeverityLevel > 0) {
            status = api->SetSessionLogSeverityLevel(sessionOptions, params->logSeverityLevel);
            if(status != NULL) {
                return releaseAndRespondErrorStatus(api, sessionOptions, status);
            }
        }
        if(params->graphOptimizationLevel != DefaultGraphOptimizationLevel) {
            status = api->SetSessionGraphOptimizationLevel(sessionOptions, params->graphOptimizationLevel);
            if(status != NULL) {
                return releaseAndRespondErrorStatus(api, sessionOptions, status);
            }
        }
        if(params->intraOpNumThreads != DefaultIntraOpNumThreads) {
            status = api->SetIntraOpNumThreads(sessionOptions, params->intraOpNumThreads);
            if(status != NULL) {
                return releaseAndRespondErrorStatus(api, sessionOptions, status);
            }
        }
        if(params->interOpNumThreads != DefaultInterOpNumThreads) {
            status = api->SetInterOpNumThreads(sessionOptions, params->interOpNumThreads);
            if(status != NULL) {
                return releaseAndRespondErrorStatus(api, sessionOptions, status);
            }
        }
        if(params->numCustomOpDomains > 0 && params->customOpDomains != NULL) {
            for(int i = 0; i < params->numCustomOpDomains; i++) {
                status = api->AddCustomOpDomain(sessionOptions, params->customOpDomains[i]);
                if(status != NULL) {
                    return releaseAndRespondErrorStatus(api, sessionOptions, status);
                }
            }
        }

        OrtCreateSessionOptionsResponse response;
        response.sessionOptions = sessionOptions;
        response.status = NULL;
        return response;
	}

	OrtCreateSessionOptionsResponse releaseAndRespondErrorStatus(OrtApi *api, OrtSessionOptions *sessionOptions, OrtStatus *status) {
	    api->ReleaseSessionOptions(sessionOptions);
	    return respondErrorStatus(status);
	}

    OrtCreateSessionOptionsResponse respondErrorStatus(OrtStatus *status) {
	    OrtCreateSessionOptionsResponse response;

	    response.status = status;
	    response.sessionOptions = NULL;

	    return response;
	}
}