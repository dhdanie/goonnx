package ort

/*
#include <onnxruntime_c_api.h>
#include "allocator.h"
*/
import "C"

type allocator struct {
	a *C.OrtAllocator
}

func newAllocatorWithDefaultOptions() (*allocator, error) {
	response := C.getAllocatorWithDefaultOptions(ortApi.ort)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	return &allocator{a: response.allocator}, nil
}
