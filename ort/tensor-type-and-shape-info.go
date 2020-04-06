package ort

/*
#include <onnxruntime_c_api.h>
#include "tensor-type-and-shape-info.h"
*/
import "C"

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
	dimCount    int
	cTensorInfo *C.OrtTensorTypeAndShapeInfo
}

func newTensorTypeAndShapeInfo(cTensorInfo *C.OrtTensorTypeAndShapeInfo) *tensorTypeAndShapeInfo {
	return &tensorTypeAndShapeInfo{
		dimCount:    -1,
		cTensorInfo: cTensorInfo,
	}
}

func (i *tensorTypeAndShapeInfo) GetElementType() (ONNXTensorElementDataType, error) {
	response := C.getTensorElementType(ortApi.ort, i.cTensorInfo)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return TensorElemDataTypeUndefined, err
	}

	return getONNXTensorElementDataTypeFromOrtTensorElementDataType(response.dataType), nil
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

	return i.GetDimensionsCount()
}

func (i *tensorTypeAndShapeInfo) GetDimensions() ([]int64, error) {
	numDims, err := i.GetDimensionsCount()
	if err != nil {
		return nil, err
	}

	cNumDims := C.size_t(numDims)

	resultContainer := make([]int64, int(cNumDims))
	status := C.getDimensions(ortApi.ort, i.cTensorInfo, cNumDims, (*C.int64_t)(&resultContainer[0]))
	err = ortApi.ParseStatus(status)
	if err != nil {
		return nil, err
	}

	return resultContainer, nil
}

func getONNXTensorElementDataTypeFromOrtTensorElementDataType(ortType C.ONNXTensorElementDataType) ONNXTensorElementDataType {
	return ONNXTensorElementDataType(ortType)
}
