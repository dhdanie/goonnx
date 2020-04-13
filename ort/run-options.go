package ort

/*
#include <onnxruntime_c_api.h>
#include "run-options.h"
*/
import "C"
import "unsafe"

type RunOptions struct {
	Tag               string
	LogVerbosityLevel int
	LogSeverityLevel  int
	Terminate         bool
}

type ortRunOptions struct {
	cRunOptions *C.OrtRunOptions
}

func (o *RunOptions) toOrtRunOptions() (*C.OrtRunOptions, error) {
	roParams := C.OrtCreateRunOptionsParameters{}
	if len(o.Tag) > 0 {
		roParams.tag = C.CString(o.Tag)
		defer C.free(unsafe.Pointer(roParams.tag))
	}
	roParams.logVerbosityLevel = C.int(o.LogSeverityLevel)
	roParams.logSeverityLevel = C.int(o.LogSeverityLevel)
	if o.Terminate {
		roParams.terminate = C.int(1)
	} else {
		roParams.terminate = C.int(0)
	}

	response := C.createRunOptions(ortApi.ort, &roParams)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}
	return response.runOptions, nil
}
