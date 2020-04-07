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
	GetInputNames() ([]string, error)
	GetInputTypeInfo(index int) (TypeInfo, error)
	GetInputTypeInfos() ([]TypeInfo, error)

	//GetOutputCount() (int, error)
	//GetOutputName(index int) (string, error)
	//GetOutputNames() ([]string, error)
	//GetOutputTypeInfo(index int) (TypeInfo, error)
	//GetOutputTypeInfos() ([]TypeInfo, error)

	Run(runOptions RunOptions)
	ReleaseSession()
}

type session struct {
	inputCount     int
	inputNames     []string
	inputTypeInfos []TypeInfo
	cModelPath     *C.char
	cSession       *C.OrtSession
	allocator      *allocator
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
		inputCount:     -1,
		inputNames:     nil,
		inputTypeInfos: nil,
		cModelPath:     cModelPath,
		cSession:       response.session,
		allocator:      allocator,
	}, nil
}

func (s *session) GetInputCount() (int, error) {
	if s.inputCount > -1 {
		return s.inputCount, nil
	}

	response := C.getInputCount(ortApi.ort, s.cSession)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return 0, err
	}

	s.inputCount = int(response.numNodes)
	return s.inputCount, nil
}

func (s *session) GetInputName(index int) (string, error) {
	inputNames, err := s.GetInputNames()
	if err != nil {
		return "", err
	}
	if index < 0 || index >= len(s.inputNames) {
		return "", fmt.Errorf("invalid input index %d", index)
	}
	return inputNames[index], nil
}

func (s *session) getInputName(index int) (string, error) {
	i := C.size_t(index)

	response := C.getInputName(ortApi.ort, s.cSession, i, s.allocator.a)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return "", err
	}

	name := C.GoString(response.name)
	C.free(unsafe.Pointer(response.name))

	return name, nil
}

func (s *session) GetInputNames() ([]string, error) {
	if s.inputNames != nil {
		return s.inputNames, nil
	}

	inputCount, err := s.GetInputCount()
	if err != nil {
		return nil, err
	}

	s.inputNames = make([]string, inputCount)
	for i := 0; i < inputCount; i++ {
		s.inputNames[i], err = s.getInputName(i)
		if err != nil {
			s.inputNames = nil
			return nil, err
		}
	}
	return s.inputNames, nil
}

func (s *session) GetInputTypeInfo(index int) (TypeInfo, error) {
	typeInfos, err := s.GetInputTypeInfos()
	if err != nil {
		return nil, err
	}
	if index < 0 || index >= len(typeInfos) {
		return nil, fmt.Errorf("invalid input index %d", index)
	}
	return typeInfos[index], nil
}

func (s *session) GetInputTypeInfos() ([]TypeInfo, error) {
	if s.inputTypeInfos != nil {
		return s.inputTypeInfos, nil
	}

	numInputs, err := s.GetInputCount()
	if err != nil {
		return nil, err
	}
	s.inputTypeInfos = make([]TypeInfo, numInputs)
	for i := 0; i < numInputs; i++ {
		s.inputTypeInfos[i], err = s.getInputTypeInfo(i)
		if err != nil {
			s.inputTypeInfos = nil
			return nil, err
		}
	}
	return s.inputTypeInfos, nil
}

func (s *session) getInputTypeInfo(index int) (TypeInfo, error) {
	i := C.size_t(index)

	response := C.getInputTypeInfo(ortApi.ort, s.cSession, i)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	return &inputTypeInfo{cTypeInfo: response.typeInfo}, nil
}

func (s *session) Run(runOptions RunOptions) {

}

func (s *session) ReleaseSession() {
	C.releaseSession(ortApi.ort, s.cSession)
	C.free(unsafe.Pointer(s.cModelPath))
}
