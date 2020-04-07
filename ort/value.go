package ort

/*
#include <onnxruntime_c_api.h>
#include "value.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type Value interface {
	IsTensor() (bool, error)
}

type value struct {
	cOrtValue *C.OrtValue
}

func NewTensorWithDataAsValue(memInfo MemoryInfo, inData []byte, typeInfo TensorTypeAndShapeInfo) (Value, error) {
	inLen := C.size_t(len(inData))

	sMemInfo, ok := memInfo.(*memoryInfo)
	if !ok {
		return nil, fmt.Errorf("invalid memory info type")
	}

	sTypeInfo, ok := typeInfo.(*tensorTypeAndShapeInfo)
	if !ok {
		return nil, fmt.Errorf("invalid tensor type and shape info")
	}

	dims, err := sTypeInfo.GetDimensions()
	if err != nil {
		return nil, err
	}
	dimCount, err := sTypeInfo.cGetDimensionsCount()
	if err != nil {
		return nil, err
	}
	elemType, err := sTypeInfo.cGetElementType()
	if err != nil {
		return nil, err
	}

	response := C.createTensorWithDataAsOrtValue(ortApi.ort, sMemInfo.cMemoryInfo, unsafe.Pointer(&inData[0]), inLen, (*C.int64_t)(&dims[0]), dimCount, elemType)
	err = ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	return &value{
		cOrtValue: response.value,
	}, nil
}

func (v *value) IsTensor() (bool, error) {
	response := C.isTensor(ortApi.ort, v.cOrtValue)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return false, err
	}
	if response.isTensor == 1 {
		return true, nil
	}
	return false, nil
}
