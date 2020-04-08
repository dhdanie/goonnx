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

	GetOutputCount() (int, error)
	GetOutputName(index int) (string, error)
	GetOutputNames() ([]string, error)
	GetOutputTypeInfo(index int) (TypeInfo, error)
	GetOutputTypeInfos() ([]TypeInfo, error)

	Run(runOptions RunOptions, inputValues []Value) ([]Value, error)
	ReleaseSession()

	PrintIOInfo()
}

type session struct {
	inputCount      int
	inputNames      []string
	inputTypeInfos  []TypeInfo
	outputCount     int
	outputNames     []string
	outputTypeInfos []TypeInfo
	cModelPath      *C.char
	cSession        *C.OrtSession
	allocator       *allocator
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
		inputCount:      -1,
		inputNames:      nil,
		inputTypeInfos:  nil,
		outputCount:     -1,
		outputNames:     nil,
		outputTypeInfos: nil,
		cModelPath:      cModelPath,
		cSession:        response.session,
		allocator:       allocator,
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

	return &typeInfo{cTypeInfo: response.typeInfo}, nil
}

func (s *session) Run(runOpts RunOptions, inputValues []Value) ([]Value, error) {
	sRunOptions, ok := runOpts.(*runOptions)
	if !ok {
		return nil, fmt.Errorf("invalid run options type")
	}

	inputNames, err := s.GetInputNames()
	if err != nil {
		return nil, err
	}
	cInputNames := stringsToCharArrayPtr(inputNames)
	defer freeCStrings(cInputNames)

	outputNames, err := s.GetOutputNames()
	if err != nil {
		return nil, err
	}
	cOutputNames := stringsToCharArrayPtr(outputNames)
	defer freeCStrings(cOutputNames)

	cInputValues, err := valuesToOrtValueArray(inputValues)
	if err != nil {
		return nil, err
	}
	inLen := C.size_t(len(inputValues))
	outNamesLen := C.size_t(len(outputNames))

	response := C.run(ortApi.ort, s.cSession, sRunOptions.cRunOptions, &cInputNames[0], &cInputValues[0], inLen, &cOutputNames[0], outNamesLen)
	err = ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	return s.outputsToValueSlice(response.output)
}

func (s *session) outputsToValueSlice(outputs *C.OrtValue) ([]Value, error) {
	typeInfo, err := s.GetOutputTypeInfo(0)
	if err != nil {
		return nil, err
	}

	tensorInfo, err := typeInfo.ToTensorInfo()
	if err != nil {
		return nil, err
	}

	return []Value{
		&value{
			typeInfo:  tensorInfo,
			cOrtValue: outputs,
		},
	}, nil
}

func stringsToCharArrayPtr(in []string) []*C.char {
	cStrings := make([]*C.char, len(in))
	for i, inVal := range in {
		cStrings[i] = C.CString(inVal)
	}
	return cStrings
}

func freeCStrings(in []*C.char) {
	for i := 0; i < len(in); i++ {
		C.free(unsafe.Pointer(in[i]))
	}
}

func valuesToOrtValueArray(in []Value) ([]*C.OrtValue, error) {
	ortVals := make([]*C.OrtValue, len(in))
	for i, inVal := range in {
		sValue, ok := inVal.(*value)
		if !ok {
			return nil, fmt.Errorf("invalid Value type")
		}
		ortVals[i] = sValue.cOrtValue
	}
	return ortVals, nil
}

func (s *session) ReleaseSession() {
	C.releaseSession(ortApi.ort, s.cSession)
	C.free(unsafe.Pointer(s.cModelPath))
}

func (s *session) GetOutputCount() (int, error) {
	if s.outputCount > -1 {
		return s.outputCount, nil
	}

	response := C.getOutputCount(ortApi.ort, s.cSession)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return 0, err
	}

	s.outputCount = int(response.numNodes)
	return s.outputCount, nil
}

func (s *session) GetOutputName(index int) (string, error) {
	outputNames, err := s.GetOutputNames()
	if err != nil {
		return "", err
	}
	if index < 0 || index >= len(s.outputNames) {
		return "", fmt.Errorf("invalid output index %d", index)
	}
	return outputNames[index], nil
}

func (s *session) getOutputName(index int) (string, error) {
	i := C.size_t(index)

	response := C.getOutputName(ortApi.ort, s.cSession, i, s.allocator.a)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return "", err
	}

	name := C.GoString(response.name)
	C.free(unsafe.Pointer(response.name))

	return name, nil
}

func (s *session) GetOutputNames() ([]string, error) {
	if s.outputNames != nil {
		return s.outputNames, nil
	}

	outputCount, err := s.GetOutputCount()
	if err != nil {
		return nil, err
	}

	s.outputNames = make([]string, outputCount)
	for i := 0; i < outputCount; i++ {
		s.outputNames[i], err = s.getOutputName(i)
		if err != nil {
			s.outputNames = nil
			return nil, err
		}
	}
	return s.outputNames, nil
}

func (s *session) GetOutputTypeInfo(index int) (TypeInfo, error) {
	typeInfos, err := s.GetOutputTypeInfos()
	if err != nil {
		return nil, err
	}
	if index < 0 || index >= len(typeInfos) {
		return nil, fmt.Errorf("invalid output index %d", index)
	}
	return typeInfos[index], nil
}

func (s *session) GetOutputTypeInfos() ([]TypeInfo, error) {
	if s.outputTypeInfos != nil {
		return s.outputTypeInfos, nil
	}

	numOutputs, err := s.GetOutputCount()
	if err != nil {
		return nil, err
	}
	s.outputTypeInfos = make([]TypeInfo, numOutputs)
	for i := 0; i < numOutputs; i++ {
		s.outputTypeInfos[i], err = s.getOutputTypeInfo(i)
		if err != nil {
			s.outputTypeInfos = nil
			return nil, err
		}
	}
	return s.outputTypeInfos, nil
}

func (s *session) getOutputTypeInfo(index int) (TypeInfo, error) {
	i := C.size_t(index)

	response := C.getOutputTypeInfo(ortApi.ort, s.cSession, i)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	return &typeInfo{cTypeInfo: response.typeInfo}, nil
}

func (s *session) PrintIOInfo() {
	fmt.Printf("*******************************\n")
	fmt.Printf("*** Session I/O Information ***\n")
	fmt.Printf("*******************************\n")
	inCount, err := s.GetInputCount()
	if err != nil {
		fmt.Printf("Error retrieving input count - %s\n", err.Error())
	} else {
		fmt.Printf("Number of inputs: %d\n", inCount)
		for i := 0; i < inCount; i++ {
			fmt.Printf("Input %d:\n", i)
			name, err := s.GetInputName(i)
			if err != nil {
				fmt.Printf("  Error retrieving name\n")
			} else {
				fmt.Printf("  Name: %s\n", name)
			}
			typeInfo, err := s.GetInputTypeInfo(i)
			if err != nil {
				fmt.Printf("  Error retrieving type info\n")
			} else {
				s.printTensorTypeInfo(typeInfo)
			}
		}
	}
	outCount, err := s.GetOutputCount()
	if err != nil {
		fmt.Printf("Error retrieving output count - %s\n", err.Error())
	} else {
		fmt.Printf("Number of outputs: %d\n", outCount)
		for i := 0; i < outCount; i++ {
			fmt.Printf("Output %d:\n", i)
			name, err := s.GetOutputName(i)
			if err != nil {
				fmt.Printf("  Error retrieving name\n")
			} else {
				fmt.Printf("  Name: %s\n", name)
			}
			typeInfo, err := s.GetOutputTypeInfo(i)
			if err != nil {
				fmt.Printf("  Error retrieving type info\n")
			} else {
				s.printTensorTypeInfo(typeInfo)
			}
		}
	}
}

func (s *session) printTensorTypeInfo(typeInfo TypeInfo) {
	tensorInfo, err := typeInfo.ToTensorInfo()
	if err != nil {
		fmt.Printf("  Error converting type info to tensor info\n")
	} else {
		onnxElementType, err := tensorInfo.GetElementType()
		if err != nil {
			fmt.Printf("  Error retrieving element type\n")
		} else {
			fmt.Printf("  Element Type: %d\n", onnxElementType)
		}
		dimsCount, err := tensorInfo.GetDimensionsCount()
		if err != nil {
			fmt.Printf("  Error retrieving dimensions count\n")
		} else {
			fmt.Printf("  Dimensions Count: %d\n", dimsCount)
		}
		dims, err := tensorInfo.GetDimensions()
		if err != nil {
			fmt.Printf("  Error retrieving dimensions\n")
		} else {
			for j, dim := range dims {
				fmt.Printf("  dim %d size: %d\n", j, dim)
			}
		}
	}
}
