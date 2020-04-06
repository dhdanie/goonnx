package ort

/*
#include <onnxruntime_c_api.h>
#include "session-options.h"
*/
import "C"
import "fmt"

type GraphOptimizationLevel int

const (
	GraphOptLevelDisableAll     GraphOptimizationLevel = 0
	GraphOptLevelEnableBasic    GraphOptimizationLevel = 1
	GraphOptLevelEnableExtended GraphOptimizationLevel = 2
	GraphOptLevelEnableAll      GraphOptimizationLevel = 99
)

type SessionOptions interface {
	SetIntraOpNumThreads(numThreads int) error
	SetSessionGraphOptimizationLevel(graphOptimizationLevel GraphOptimizationLevel) error
	ReleaseSessionOptions()
}

type sessionOptions struct {
	intraOpNumThreads      int
	graphOptimizationLevel GraphOptimizationLevel
	cSessionOptions        *C.OrtSessionOptions
}

func NewSessionOptions() (SessionOptions, error) {
	response := C.createSessionOptions(ortApi.ort)
	errorMsg := ortApi.ParseStatus(response.status)
	if errorMsg != nil {
		return nil, fmt.Errorf("%s", errorMsg)
	}

	return &sessionOptions{cSessionOptions: response.sessionOptions}, nil
}

func (o *sessionOptions) SetIntraOpNumThreads(numThreads int) error {
	o.intraOpNumThreads = numThreads

	cNumThreads := C.int(numThreads)
	status := C.setIntraOpNumThreads(ortApi.ort, o.cSessionOptions, cNumThreads)

	return ortApi.ParseStatus(status)
}

func (o *sessionOptions) SetSessionGraphOptimizationLevel(graphOptimizationLevel GraphOptimizationLevel) error {
	o.graphOptimizationLevel = graphOptimizationLevel

	level, err := getOrtSessionGraphOptimizationLevelForGraphOptimizationLevel(graphOptimizationLevel)
	if err != nil {
		return err
	}

	status := C.setSessionGraphOptimizationLevel(ortApi.ort, o.cSessionOptions, level)

	return ortApi.ParseStatus(status)
}

func (o *sessionOptions) ReleaseSessionOptions() {
	C.releaseSessionOptions(ortApi.ort, o.cSessionOptions)
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
