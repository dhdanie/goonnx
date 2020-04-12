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

type SessionOptions interface {
	SetOptimizedModelFilePath(optimizedModelFilepath string)
	SetSessionExecutionMode(executionMode ExecutionMode)
	EnableProfiling(profileFilePrefix string)
	DisableProfiling()
	EnableMemPattern()
	DisableMemPattern()
	EnableCPUMemArena()
	DisableCPUMemArena()
	SetSessionLogID(logID string)
	SetSessionLogVerbosityLevel(level int)
	SetSessionLogSeverityLevel(level int)
	SetSessionGraphOptimizationLevel(graphOptimizationLevel GraphOptimizationLevel)
	SetIntraOpNumThreads(numThreads int)
	SetInterOpNumThreads(numThreads int)
	AddCustomOpDomain(opDomain CustomOpDomain)
	Clone() SessionOptions

	toOrtSessionOptions() (*ortSessionOptions, error)
}

type ortSessionOptions struct {
	cOrtSessionOptions *C.OrtSessionOptions
}

type sessionOptions struct {
	optimizedModelFilePath string
	executionMode          ExecutionMode
	profilingEnabled       bool
	profileFilePrefix      string
	memPatternEnabled      bool
	cpuMemArenaEnabled     bool
	sessionLogID           string
	logVerbosityLevel      int
	logSeverityLevel       int
	graphOptimizationLevel GraphOptimizationLevel
	intraOpNumThreads      int
	interOpNumThreads      int
	customOpDomains        []CustomOpDomain
}

func NewSessionOptions() SessionOptions {
	return &sessionOptions{
		optimizedModelFilePath: "",
		executionMode:          ExecutionModeSequential,
		profilingEnabled:       false,
		profileFilePrefix:      "",
		memPatternEnabled:      false,
		cpuMemArenaEnabled:     false,
		sessionLogID:           "",
		logVerbosityLevel:      0,
		logSeverityLevel:       0,
		graphOptimizationLevel: GraphOptLevelEnableAll,
		intraOpNumThreads:      0,
		interOpNumThreads:      0,
		customOpDomains:        nil,
	}
}

func (o *sessionOptions) SetOptimizedModelFilePath(optimizedModelFilepath string) {
	o.optimizedModelFilePath = optimizedModelFilepath
}

func (o *sessionOptions) SetSessionExecutionMode(executionMode ExecutionMode) {
	o.executionMode = executionMode
}

func (o *sessionOptions) EnableProfiling(profileFilePrefix string) {
	o.profilingEnabled = true
	o.profileFilePrefix = profileFilePrefix
}

func (o *sessionOptions) DisableProfiling() {
	o.profilingEnabled = false
	o.profileFilePrefix = ""
}

func (o *sessionOptions) EnableMemPattern() {
	o.memPatternEnabled = true
}

func (o *sessionOptions) DisableMemPattern() {
	o.memPatternEnabled = false
}

func (o *sessionOptions) EnableCPUMemArena() {
	o.cpuMemArenaEnabled = true
}

func (o *sessionOptions) DisableCPUMemArena() {
	o.cpuMemArenaEnabled = false
}

func (o *sessionOptions) SetSessionLogID(logID string) {
	o.sessionLogID = logID
}

func (o *sessionOptions) SetSessionLogVerbosityLevel(level int) {
	o.logVerbosityLevel = level
}

func (o *sessionOptions) SetSessionLogSeverityLevel(level int) {
	o.logSeverityLevel = level
}

func (o *sessionOptions) SetSessionGraphOptimizationLevel(graphOptimizationLevel GraphOptimizationLevel) {
	o.graphOptimizationLevel = graphOptimizationLevel
}

func (o *sessionOptions) SetIntraOpNumThreads(numThreads int) {
	o.intraOpNumThreads = numThreads
}

func (o *sessionOptions) SetInterOpNumThreads(numThreads int) {
	o.interOpNumThreads = numThreads
}

func (o *sessionOptions) AddCustomOpDomain(opDomain CustomOpDomain) {
	o.customOpDomains = append(o.customOpDomains, opDomain)
}

func (o *sessionOptions) Clone() SessionOptions {
	return &sessionOptions{
		optimizedModelFilePath: o.optimizedModelFilePath,
		executionMode:          o.executionMode,
		profilingEnabled:       o.profilingEnabled,
		profileFilePrefix:      o.profileFilePrefix,
		memPatternEnabled:      o.memPatternEnabled,
		cpuMemArenaEnabled:     o.cpuMemArenaEnabled,
		sessionLogID:           o.sessionLogID,
		logVerbosityLevel:      o.logVerbosityLevel,
		logSeverityLevel:       o.logSeverityLevel,
		graphOptimizationLevel: o.graphOptimizationLevel,
		intraOpNumThreads:      o.intraOpNumThreads,
		interOpNumThreads:      o.interOpNumThreads,
		customOpDomains:        nil,
	}
}

func (o *sessionOptions) toOrtSessionOptions() (*ortSessionOptions, error) {
	var err error

	soParams := C.OrtCreateSessionOptionsParams{}
	if len(o.optimizedModelFilePath) > 0 {
		soParams.optimizedModelFilePath = C.CString(o.optimizedModelFilePath)
		defer C.free(unsafe.Pointer(soParams.optimizedModelFilePath))
	} else {
		soParams.optimizedModelFilePath = nil
	}
	soParams.executionMode, err = getOrtExecutionModeForExecutionMode(o.executionMode)
	if err != nil {
		soParams.executionMode, _ = getOrtExecutionModeForExecutionMode(DefaultExecutionMode)
		err = nil
	}
	if o.profilingEnabled && len(o.profileFilePrefix) > 0 {
		soParams.profilingEnabled = C.int(1)
		soParams.profileFilePrefix = C.CString(o.profileFilePrefix)
		defer C.free(unsafe.Pointer(soParams.profileFilePrefix))
	} else {
		soParams.profilingEnabled = C.int(0)
		soParams.profileFilePrefix = nil
	}
	if o.memPatternEnabled {
		soParams.memPatternEnabled = C.int(1)
	} else {
		soParams.memPatternEnabled = C.int(0)
	}
	if o.cpuMemArenaEnabled {
		soParams.cpuMemArenaEnabled = C.int(1)
	} else {
		soParams.cpuMemArenaEnabled = C.int(0)
	}
	if len(o.sessionLogID) > 0 {
		soParams.logId = C.CString(o.sessionLogID)
		defer C.free(unsafe.Pointer(soParams.logId))
	} else {
		soParams.logId = nil
	}
	soParams.logVerbosityLevel = C.int(o.logVerbosityLevel)
	soParams.logSeverityLevel = C.int(o.logSeverityLevel)
	soParams.graphOptimizationLevel, err = getOrtSessionGraphOptimizationLevelForGraphOptimizationLevel(o.graphOptimizationLevel)
	if err != nil {
		soParams.graphOptimizationLevel, _ = getOrtSessionGraphOptimizationLevelForGraphOptimizationLevel(DefaultGraphOptLevel)
		err = nil
	}
	soParams.intraOpNumThreads = C.int(o.intraOpNumThreads)
	soParams.interOpNumThreads = C.int(o.interOpNumThreads)
	soParams.numCustomOpDomains = C.int(len(o.customOpDomains))
	if len(o.customOpDomains) > 0 {
		cCustomOpDomains := make([]*C.OrtCustomOpDomain, len(o.customOpDomains))
		for i, customOpDomain := range o.customOpDomains {
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
