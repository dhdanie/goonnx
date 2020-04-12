package ort

/*
#include <onnxruntime_c_api.h>
#include "type-info.h"
*/
import "C"
import "fmt"

type ONNXType int

const (
	TypeUnknown      ONNXType = 0
	TypeTensor       ONNXType = 1
	TypeSequence     ONNXType = 2
	TypeMap          ONNXType = 3
	TypeOpaque       ONNXType = 4
	TypeSparseTensor ONNXType = 5
)

type TypeInfo interface {
	ToTensorInfo() (TensorTypeAndShapeInfo, error)
	ReleaseTypeInfo()
}

type typeInfo struct {
	released  bool
	cTypeInfo *C.OrtTypeInfo
}

func newTypeInfo(cTypeInfo *C.OrtTypeInfo) TypeInfo {
	return &typeInfo{
		released:  false,
		cTypeInfo: cTypeInfo,
	}
}

func (i *typeInfo) ToTensorInfo() (TensorTypeAndShapeInfo, error) {
	if i.cTypeInfo == nil {
		return nil, fmt.Errorf("TypeInfo incorrectly instantiated")
	}

	response := C.castTypeInfoToTensorInfo(ortApi.ort, i.cTypeInfo)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	return newTensorTypeAndShapeInfo(response.tensorInfo), nil
}

func (i *typeInfo) ReleaseTypeInfo() {
	if !i.released {
		C.releaseTypeInfo(ortApi.ort, i.cTypeInfo)
		i.cTypeInfo = nil
		i.released = true
	}
}
