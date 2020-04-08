package ort

/*
#include <onnxruntime_c_api.h>
#include "value.h"
*/
import "C"
import (
	"fmt"
	"reflect"
	"unsafe"
)

type Value interface {
	IsTensor() (bool, error)
	GetTensorMutableFloatData() ([]float32, error)
}

type value struct {
	typeInfo  TensorTypeAndShapeInfo
	cOrtValue *C.OrtValue
}

func NewTensorWithFloatDataAsValue(memInfo MemoryInfo, inData []float32, typeInfo TensorTypeAndShapeInfo) (Value, error) {
	actInLen := uintptr(len(inData)) * reflect.TypeOf(inData[0]).Size()
	inLen := C.size_t(actInLen)

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
		typeInfo:  typeInfo,
		cOrtValue: response.value,
	}, nil
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

func (v *value) GetTensorMutableFloatData() ([]float32, error) {
	response := C.getTensorMutableFloatData(ortApi.ort, v.cOrtValue)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	len, err := v.calcDataSize()
	if err != nil {
		return nil, err
	}

	var data []float32
	sliceHeader := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	sliceHeader.Cap = int(len)
	sliceHeader.Len = int(len)
	sliceHeader.Data = uintptr(unsafe.Pointer(response.out))

	output := append([]float32(nil), data...)

	return output, nil
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

func (v *value) calcDataSize() (int64, error) {
	dims, err := v.typeInfo.GetDimensions()
	if err != nil {
		return -1, err
	}

	var total int64 = 1
	for _, dim := range dims {
		total = total * dim
	}
	return total, nil
}
