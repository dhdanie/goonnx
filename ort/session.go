package ort

/*
#include <onnxruntime_c_api.h>
#include "session.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type Session interface {
	GetInputCount() (int, error)
	GetInputName(index int) (string, error)
	GetInputTypeInfo(index int) (InputTypeInfo, error)
	ReleaseSession()
}

type session struct {
	cModelPath *C.char
	cSession   *C.OrtSession
	allocator  *allocator
}

func NewSession(env Environment, modelPath string, sessionOpts SessionOptions) (Session, error) {
	cModelPath := C.CString(modelPath)

	e, ok := env.(*environment)
	if !ok {
		return nil, fmt.Errorf("invalid Environment type")
	}

	so, ok := sessionOpts.(*sessionOptions)
	if !ok {
		return nil, fmt.Errorf("invalid SessionOptions type")
	}

	response := C.createSession(ortApi.ort, e.env, cModelPath, so.cSessionOptions)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	allocator, err := newAllocatorWithDefaultOptions()
	if err != nil {
		return nil, err
	}

	return &session{
		cModelPath: cModelPath,
		cSession:   response.session,
		allocator:  allocator,
	}, nil
}

func (s *session) GetInputCount() (int, error) {
	response := C.getInputCount(ortApi.ort, s.cSession)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return 0, err
	}

	return int(response.numInputNodes), nil
}

func (s *session) GetInputName(index int) (string, error) {
	i := C.size_t(index)

	response := C.getInputName(ortApi.ort, s.cSession, i, s.allocator.a)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return "", err
	}

	name := C.GoString(response.inputName)
	C.free(unsafe.Pointer(response.inputName))

	return name, nil
}

func (s *session) GetInputTypeInfo(index int) (InputTypeInfo, error) {
	i := C.size_t(index)

	response := C.getInputTypeInfo(ortApi.ort, s.cSession, i)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	return &inputTypeInfo{cTypeInfo: response.typeInfo}, nil
}

func (s *session) ReleaseSession() {
	C.releaseSession(ortApi.ort, s.cSession)
	C.free(unsafe.Pointer(s.cModelPath))
}
