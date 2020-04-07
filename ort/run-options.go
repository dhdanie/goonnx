package ort

/*
#include <onnxruntime_c_api.h>
#include "session.h"
*/
import "C"

type RunOptions interface {
}

type runOptions struct {
	cRunOptions *C.OrtRunOptions
}

func NewRunOptions() RunOptions {
	return &runOptions{}
}
