package ort

/*
#include <onnxruntime_c_api.h>
#include "type-info.h"
*/
import "C"

type TypeInfo interface {
	ToTensorInfo() (TensorTypeAndShapeInfo, error)
	ReleaseTypeInfo()
}

type inputTypeInfo struct {
	cTypeInfo *C.OrtTypeInfo
}

func (i *inputTypeInfo) ToTensorInfo() (TensorTypeAndShapeInfo, error) {
	response := C.castTypeInfoToTensorInfo(ortApi.ort, i.cTypeInfo)
	err := ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	return newTensorTypeAndShapeInfo(response.tensorInfo), nil
}

func (i *inputTypeInfo) ReleaseTypeInfo() {
	C.releaseTypeInfo(ortApi.ort, i.cTypeInfo)
}
