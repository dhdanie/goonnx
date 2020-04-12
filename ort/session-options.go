package ort

/*
#include <onnxruntime_c_api.h>
#include "session-options.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type ExecutionMode int
type GraphOptimizationLevel int

const (
	ExecutionModeSequential ExecutionMode = 0
	ExecutionModeParallel   ExecutionMode = 1
)
const (
	GraphOptLevelDisableAll     GraphOptimizationLevel = 0
	GraphOptLevelEnableBasic    GraphOptimizationLevel = 1
	GraphOptLevelEnableExtended GraphOptimizationLevel = 2
	GraphOptLevelEnableAll      GraphOptimizationLevel = 99
)

const DefaultExecutionMode ExecutionMode = ExecutionModeSequential
const DefaultGraphOptLevel GraphOptimizationLevel = GraphOptLevelEnableAll

type ortSessionOptions struct {
	cOrtSessionOptions *C.OrtSessionOptions
}

type SessionOptions struct {
	OptimizedModelFilePath string
	ExecutionMode          ExecutionMode
	ProfilingEnabled       bool
	ProfileFilePrefix      string
	MemPatternEnabled      bool
	CPUMemArenaEnabled     bool
	SessionLogID           string
	LogVerbosityLevel      int
	LogSeverityLevel       int
	GraphOptimizationLevel GraphOptimizationLevel
	IntraOpNumThreads      int
	InterOpNumThreads      int
	CustomOpDomains        []CustomOpDomain
}

func (o *SessionOptions) Clone() *SessionOptions {
	return &SessionOptions{
		OptimizedModelFilePath: o.OptimizedModelFilePath,
		ExecutionMode:          o.ExecutionMode,
		ProfilingEnabled:       o.ProfilingEnabled,
		ProfileFilePrefix:      o.ProfileFilePrefix,
		MemPatternEnabled:      o.MemPatternEnabled,
		CPUMemArenaEnabled:     o.CPUMemArenaEnabled,
		SessionLogID:           o.SessionLogID,
		LogVerbosityLevel:      o.LogVerbosityLevel,
		LogSeverityLevel:       o.LogSeverityLevel,
		GraphOptimizationLevel: o.GraphOptimizationLevel,
		IntraOpNumThreads:      o.IntraOpNumThreads,
		InterOpNumThreads:      o.InterOpNumThreads,
		CustomOpDomains:        nil,
	}
}

func (o *SessionOptions) toOrtSessionOptions() (*ortSessionOptions, error) {
	var err error

	soParams := C.OrtCreateSessionOptionsParams{}
	if len(o.OptimizedModelFilePath) > 0 {
		soParams.optimizedModelFilePath = C.CString(o.OptimizedModelFilePath)
		defer C.free(unsafe.Pointer(soParams.optimizedModelFilePath))
	} else {
		soParams.optimizedModelFilePath = nil
	}
	soParams.executionMode, err = getOrtExecutionModeForExecutionMode(o.ExecutionMode)
	if err != nil {
		soParams.executionMode, _ = getOrtExecutionModeForExecutionMode(DefaultExecutionMode)
		err = nil
	}
	if o.ProfilingEnabled && len(o.ProfileFilePrefix) > 0 {
		soParams.profilingEnabled = C.int(1)
		soParams.profileFilePrefix = C.CString(o.ProfileFilePrefix)
		defer C.free(unsafe.Pointer(soParams.profileFilePrefix))
	} else {
		soParams.profilingEnabled = C.int(0)
		soParams.profileFilePrefix = nil
	}
	if o.MemPatternEnabled {
		soParams.memPatternEnabled = C.int(1)
	} else {
		soParams.memPatternEnabled = C.int(0)
	}
	if o.CPUMemArenaEnabled {
		soParams.cpuMemArenaEnabled = C.int(1)
	} else {
		soParams.cpuMemArenaEnabled = C.int(0)
	}
	if len(o.SessionLogID) > 0 {
		soParams.logId = C.CString(o.SessionLogID)
		defer C.free(unsafe.Pointer(soParams.logId))
	} else {
		soParams.logId = nil
	}
	soParams.logVerbosityLevel = C.int(o.LogVerbosityLevel)
	soParams.logSeverityLevel = C.int(o.LogSeverityLevel)
	soParams.graphOptimizationLevel, err = getOrtSessionGraphOptimizationLevelForGraphOptimizationLevel(o.GraphOptimizationLevel)
	if err != nil {
		soParams.graphOptimizationLevel, _ = getOrtSessionGraphOptimizationLevelForGraphOptimizationLevel(DefaultGraphOptLevel)
		err = nil
	}
	soParams.intraOpNumThreads = C.int(o.IntraOpNumThreads)
	soParams.interOpNumThreads = C.int(o.InterOpNumThreads)
	soParams.numCustomOpDomains = C.int(len(o.CustomOpDomains))
	if len(o.CustomOpDomains) > 0 {
		cCustomOpDomains := make([]*C.OrtCustomOpDomain, len(o.CustomOpDomains))
		for i, customOpDomain := range o.CustomOpDomains {
			cCustomOpDomains[i] = customOpDomain.toCCustomOpDomain()
		}
		soParams.customOpDomains = &cCustomOpDomains[0]
	} else {
		soParams.customOpDomains = nil
	}

	response := C.createSessionOptions(ortApi.ort, &soParams)
	err = ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}
	return &ortSessionOptions{cOrtSessionOptions: response.sessionOptions}, nil
}

func getOrtSessionGraphOptimizationLevelForGraphOptimizationLevel(level GraphOptimizationLevel) (C.GraphOptimizationLevel, error) {
	switch level {
	case GraphOptLevelDisableAll:
		return C.ORT_DISABLE_ALL, nil
	case GraphOptLevelEnableBasic:
		return C.ORT_ENABLE_BASIC, nil
	case GraphOptLevelEnableExtended:
		return C.ORT_ENABLE_EXTENDED, nil
	case GraphOptLevelEnableAll:
		return C.ORT_ENABLE_ALL, nil
	}
	return 0, fmt.Errorf("invalid graph optimization level %d", level)
}

func getOrtExecutionModeForExecutionMode(executionMode ExecutionMode) (C.ExecutionMode, error) {
	switch executionMode {
	case ExecutionModeSequential:
		return C.ORT_SEQUENTIAL, nil
	case ExecutionModeParallel:
		return C.ORT_PARALLEL, nil
	}
	return 0, fmt.Errorf("invalid execution mode %d", executionMode)
}
