package ort

/*
#include <onnxruntime_c_api.h>
*/
import "C"

type CustomOpDomain interface {
	//TODO
	AddCustomOp(op CustomOp) error
	toCCustomOpDomain() *C.OrtCustomOpDomain
}

func CreateCustomOpDomain(domain string) (CustomOpDomain, error) {
	//TODO
	return nil, nil
}
