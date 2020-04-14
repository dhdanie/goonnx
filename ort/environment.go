package ort

/*
#include <onnxruntime_c_api.h>
#include "environment.h"
*/
import "C"
import (
	"fmt"
	"sync"
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

type CustomLogger func(severity LoggingLevel, category string, codeLocation string, message string)
type cCustomLogger func(params unsafe.Pointer, severity C.OrtLoggingLevel, category *C.char, logId *C.char, codeLocation *C.char, message *C.char)

type Environment interface {
	ReleaseEnvironment()
}

type environment struct {
	env    *C.OrtEnv
	cLogId *C.char
	logger CustomLogger
}

var mu sync.Mutex
var customLoggers = make(map[string]cCustomLogger)

func NewEnvironment(loggingLevel LoggingLevel, logId string) (Environment, error) {
	logLevel, err := getOrtLoggingLevelForLoggingLevel(loggingLevel)
	if err != nil {
		return nil, err
	}

	cLogId := C.CString(logId)

	response := C.createEnv(ortApi.ort, logLevel, cLogId)
	err = ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	return &environment{
		env:    response.env,
		cLogId: cLogId,
		logger: nil,
	}, nil
}

func NewEnvironmentWithCustomLogger(loggingLevel LoggingLevel, logId string, logger CustomLogger) (Environment, error) {
	logLevel, err := getOrtLoggingLevelForLoggingLevel(loggingLevel)
	if err != nil {
		return nil, err
	}

	cLogId := C.CString(logId)

	response := C.createEnvWithCustomLogger(ortApi.ort, nil, logLevel, cLogId)
	err = ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	env := &environment{
		env:    response.env,
		cLogId: cLogId,
		logger: logger,
	}
	register(logId, env)
	return env, nil
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

func getLoggingLeveForOrtLoggingLevel(ortLoggingLevel C.OrtLoggingLevel) (LoggingLevel, error) {
	switch ortLoggingLevel {
	case C.ORT_LOGGING_LEVEL_VERBOSE:
		return LoggingLevelVerbose, nil
	case C.ORT_LOGGING_LEVEL_INFO:
		return LoggingLevelInfo, nil
	case C.ORT_LOGGING_LEVEL_WARNING:
		return LoggingLevelWarning, nil
	case C.ORT_LOGGING_LEVEL_ERROR:
		return LoggingLevelError, nil
	case C.ORT_LOGGING_LEVEL_FATAL:
		return LoggingLevelFatal, nil
	}
	return 0, fmt.Errorf("invalid ORT logging level %d", int(ortLoggingLevel))
}

func (e *environment) logCustom(params unsafe.Pointer, severity C.OrtLoggingLevel, category *C.char, logId *C.char, codeLocation *C.char, message *C.char) {
	if e.logger != nil {
		level, err := getLoggingLeveForOrtLoggingLevel(severity)
		if err != nil {
			level = LoggingLevelError
		}
		cat := C.GoString(category)
		loc := C.GoString(codeLocation)
		msg := C.GoString(message)

		e.logger(level, cat, loc, msg)
	}
}

func (e *environment) ReleaseEnvironment() {
	C.releaseEnv(ortApi.ort, e.env)
	C.free(unsafe.Pointer(e.cLogId))
}

//export logCustom
func logCustom(params unsafe.Pointer, severity C.OrtLoggingLevel, category *C.char, logId *C.char, codeLocation *C.char, message *C.char) {
	sLogId := C.GoString(logId)
	f := lookup(sLogId)
	if f != nil {
		f(params, severity, category, logId, codeLocation, message)
	}
}

func register(logId string, env *environment) {
	mu.Lock()
	defer mu.Unlock()

	customLoggers[logId] = env.logCustom
}

func lookup(logId string) cCustomLogger {
	mu.Lock()
	defer mu.Unlock()

	logger := customLoggers[logId]
	if logger == nil {
		return nil
	}
	return logger
}
