package ort

/*
#include <onnxruntime_c_api.h>
#include "session.h"
*/
import "C"

type RunOptions struct {
	cRunOptions *C.OrtRunOptions
}

func (o *RunOptions) toOrtRunOptions() (*C.OrtRunOptions, error) {
	return nil, nil
}
