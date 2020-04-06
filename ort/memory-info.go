package ort
/*
#include <onnxruntime_c_api.h>
#include "memory-info.h"
 */
import "C"
import "fmt"

type AllocatorType int

const (
	AllocatorTypeInvalid AllocatorType = -1
	AllocatorTypeDevice  AllocatorType = 0
	AllocatorTypeArena   AllocatorType = 1
)

type MemType int

const (
	MemTypeCPUInput  MemType = -2
	MemTypeCPUOutput MemType = -1
	MemTypeCPU       MemType = MemTypeCPUOutput
	MemTypeDefault   MemType = 0
)

type MemoryInfo interface {
	ReleaseMemoryInfo()
}

type memoryInfo struct {
	allocatorType AllocatorType
	memType MemType
	cMemoryInfo *C.OrtMemoryInfo
}

func NewCPUMemoryInfo(allocatorType AllocatorType, memType MemType) (MemoryInfo, error) {
	cAllocatorType, err := getCAllocatorTypeForAllocatorType(allocatorType)
	if err != nil {
		return nil, err
	}
	cMemType, err := getCMemTypeForMemType(memType)
	if err != nil {
		return nil, err
	}

	response := C.createCpuMemoryInfo(ortApi.ort, cAllocatorType, cMemType)
	err = ortApi.ParseStatus(response.status)
	if err != nil {
		return nil, err
	}

	return &memoryInfo {
		allocatorType: allocatorType,
		memType: memType,
		cMemoryInfo: response.memoryInfo,
	}, nil
}

func (i *memoryInfo) ReleaseMemoryInfo() {

}

func getCAllocatorTypeForAllocatorType(allocatorType AllocatorType) (C.OrtAllocatorType, error) {
	switch allocatorType {
	case AllocatorTypeInvalid:
		return C.Invalid, nil
	case AllocatorTypeDevice:
		return C.OrtDeviceAllocator, nil
	case AllocatorTypeArena:
		return C.OrtArenaAllocator, nil
	}
	return C.Invalid, fmt.Errorf("invalid allocator type %d", allocatorType)
}

func getCMemTypeForMemType(memType MemType) (C.OrtMemType, error) {
	switch memType {
	case MemTypeCPUInput:
		return C.OrtMemTypeCPUInput, nil
	case MemTypeCPUOutput:
		return C.OrtMemTypeCPUOutput, nil
	case MemTypeDefault:
		return C.OrtMemTypeDefault, nil
	}
	return -3, fmt.Errorf("invalid memory type %d", memType)
}