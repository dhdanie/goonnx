package ort

/*
#cgo LDFLAGS: -L/usr/local/lib/onnx -lonnxruntime
#include <onnxruntime_c_api.h>
#include "api.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type api struct {
	ort *C.OrtApi
}

var ortApi = newApi()

func newApi() *api {
	return &api{
		ort: C.getApi(),
	}
}

func (a *api) ParseStatus(status *C.OrtStatus) error {
	if status == nil {
		return nil
	}

	cMessage := C.parseStatus(a.ort, status)
	defer C.free(unsafe.Pointer(cMessage))
	var message string
	message = C.GoString(cMessage)

	return fmt.Errorf("%s", message)
}
