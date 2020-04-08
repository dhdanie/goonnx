package ort

/*
#include <onnxruntime_c_api.h>
#include "tensor-type-and-shape-info.h"
*/
import "C"
import (
	"reflect"
	"unsafe"
)

type ONNXTensorElementDataType int

const (
	TensorElemDataTypeUndefined  ONNXTensorElementDataType = 0
	TensorElemDataTypeFloat      ONNXTensorElementDataType = 1
	TensorElemDataTypeUInt8      ONNXTensorElementDataType = 2
	TensorElemDataTypeInt8       ONNXTensorElementDataType = 3
	TensorElemDataTypeUInt16     ONNXTensorElementDataType = 4
	TensorElemDataTypeInt16      ONNXTensorElementDataType = 5
	TensorElemDataTypeInt32      ONNXTensorElementDataType = 6
	TensorElemDataTypeInt64      ONNXTensorElementDataType = 7
	TensorElemDataTypeString     ONNXTensorElementDataType = 8
	TensorElemDataTypeBool       ONNXTensorElementDataType = 9
	TensorElemDataTypeFloat16    ONNXTensorElementDataType = 10
	TensorElemDataTypeDouble     ONNXTensorElementDataType = 11
	TensorElemDataTypeUInt32     ONNXTensorElementDataType = 12
	TensorElemDataTypeUInt64     ONNXTensorElementDataType = 13
	TensorElemDataTypeComplex64  ONNXTensorElementDataType = 14
	TensorElemDataTypeComplex128 ONNXTensorElementDataType = 15
	TensorElemDataTypeBFloat16   ONNXTensorElementDataType = 16
)

type TensorTypeAndShapeInfo interface {
	GetElementType() (ONNXTensorElementDataType, error)
	GetDimensionsCount() (int, error)
	GetDimensions() ([]int64, error)
}

type tensorTypeAndShapeInfo struct {
	elementType ONNXTensorElementDataType
	dimCount    int
	dims        []int64
	cTensorInfo *C.OrtTensorTypeAndShapeInfo
}

func newTensorTypeAndShapeInfo(cTensorInfo *C.OrtTensorTypeAndShapeInfo) *tensorTypeAndShapeInfo {
	return &tensorTypeAndShapeInfo{
		elementType: -1,
		dimCount:    -1,
		dims:        nil,
		cTensorInfo: cTensorInfo,
	}
}

func (i *tensorTypeAndShapeInfo) GetElementType() (ONNXTensorElementDataType, error) {
	if i.elementType > -1 {
		return i.elementType, nil
	}

	response := C.getTensorElementType(ortApi.ort, i.cTensorInfo)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return TensorElemDataTypeUndefined, err
	}

	i.elementType = getONNXTensorElementDataTypeForOrtTensorElementDataType(response.dataType)
	return i.elementType, nil
}

func (i *tensorTypeAndShapeInfo) GetDimensionsCount() (int, error) {
	if i.dimCount > -1 {
		return i.dimCount, nil
	}

	response := C.getDimensionsCount(ortApi.ort, i.cTensorInfo)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return 0, err
	}

	i.dimCount = int(response.numDims)
	return i.dimCount, nil
}

func (i *tensorTypeAndShapeInfo) GetDimensions() ([]int64, error) {
	if i.dims != nil {
		return i.dims, nil
	}

	numDims, err := i.GetDimensionsCount()
	if err != nil {
		return nil, err
	}

	cNumDims := C.size_t(numDims)

	response := C.getDimensions(ortApi.ort, i.cTensorInfo, cNumDims)
	err = ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	sliceHeader := (*reflect.SliceHeader)(unsafe.Pointer(&(i.dims)))
	sliceHeader.Cap = numDims
	sliceHeader.Len = numDims
	sliceHeader.Data = uintptr(unsafe.Pointer(response.dims))

	return i.dims, nil
}

func (i *tensorTypeAndShapeInfo) cGetDimensions() (interface{}, error) {
	dims, err := i.GetDimensions()
	if err != nil {
		return nil, err
	}

	return &dims[0], nil
}

func (i *tensorTypeAndShapeInfo) cGetDimensionsCount() (C.size_t, error) {
	numDims, err := i.GetDimensionsCount()
	if err != nil {
		return 0, nil
	}
	return C.size_t(numDims), nil
}

func (i *tensorTypeAndShapeInfo) cGetElementType() (C.ONNXTensorElementDataType, error) {
	elemType, err := i.GetElementType()
	if err != nil {
		return 0, err
	}
	return getOrtTensorElementDataTypeForONNXTensorElementDataType(elemType), nil
}

func getONNXTensorElementDataTypeForOrtTensorElementDataType(ortType C.ONNXTensorElementDataType) ONNXTensorElementDataType {
	return ONNXTensorElementDataType(ortType)
}

func getOrtTensorElementDataTypeForONNXTensorElementDataType(onnxType ONNXTensorElementDataType) C.ONNXTensorElementDataType {
	return C.ONNXTensorElementDataType(onnxType)
}
