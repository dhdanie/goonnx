package ort

/*
#include <onnxruntime_c_api.h>
#include "environment.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type LoggingLevel int

const (
	LoggingLevelVerbose LoggingLevel = 0
	LoggingLevelInfo    LoggingLevel = 1
	LoggingLevelWarning LoggingLevel = 2
	LoggingLevelError   LoggingLevel = 3
	LoggingLevelFatal   LoggingLevel = 4
)

type Environment interface {
	ReleaseEnvironment()
}

type environment struct {
	env    *C.OrtEnv
	cLogId *C.char
}

func NewEnvironment(loggingLevel LoggingLevel, logId string) (Environment, error) {
	logLevel, err := getOrtLoggingLevelForLoggingLevel(loggingLevel)
	if err != nil {
		return nil, err
	}

	cLogId := C.CString(logId)

	response := C.createEnv(ortApi.ort, logLevel, cLogId)
	errorMsg := ortApi.ParseStatus(response.status)
	if errorMsg != nil {
		return nil, fmt.Errorf("%s", errorMsg)
	}

	return &environment{
		env:    response.env,
		cLogId: cLogId,
	}, nil
}

func getOrtLoggingLevelForLoggingLevel(loggingLevel LoggingLevel) (C.OrtLoggingLevel, error) {
	switch loggingLevel {
	case LoggingLevelVerbose:
		return C.ORT_LOGGING_LEVEL_VERBOSE, nil
	case LoggingLevelInfo:
		return C.ORT_LOGGING_LEVEL_INFO, nil
	case LoggingLevelWarning:
		return C.ORT_LOGGING_LEVEL_WARNING, nil
	case LoggingLevelError:
		return C.ORT_LOGGING_LEVEL_ERROR, nil
	case LoggingLevelFatal:
		return C.ORT_LOGGING_LEVEL_FATAL, nil
	}
	return 0, fmt.Errorf("invalid logging level %d", loggingLevel)
}

func (e *environment) ReleaseEnvironment() {
	C.releaseEnv(ortApi.ort, e.env)
	C.free(unsafe.Pointer(e.cLogId))
}
