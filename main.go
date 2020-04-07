package main

import (
	"encoding/binary"
	"fmt"
	"github.com/dhdanie/goonnx/ort"
	"math"
)

func main() {
	env, err := ort.NewEnvironment(ort.LoggingLevelWarning, "abcde")
	if err != nil {
		panic(err)
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		panic(err)
	}
	err = opts.SetIntraOpNumThreads(1)
	if err != nil {
		panic(err)
	}
	err = opts.SetSessionGraphOptimizationLevel(ort.GraphOptLevelEnableBasic)
	if err != nil {
		panic(err)
	}

	session, err := ort.NewSession(env, "squeezenet/model.onnx", opts)
	if err != nil {
		panic(err)
	}

	numNodes, err := session.GetInputCount()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Number of inputs = %d\n", numNodes)

	var tensorInfo ort.TensorTypeAndShapeInfo
	for i := 0; i < numNodes; i++ {
		inputName, err := session.GetInputName(i)
		if err != nil {
			panic(err)
		}
		fmt.Printf("Input %d : name=%s\n", i, inputName)

		typeInfo, err := session.GetInputTypeInfo(i)
		if err != nil {
			panic(err)
		}
		tensorInfo, err = typeInfo.ToTensorInfo()
		if err != nil {
			panic(err)
		}

		elemType, err := tensorInfo.GetElementType()
		if err != nil {
			panic(err)
		}
		fmt.Printf("Input %d : type=%d\n", i, elemType)

		numDims, err := tensorInfo.GetDimensionsCount()
		if err != nil {
			panic(err)
		}

		dims, err := tensorInfo.GetDimensions()
		if err != nil {
			panic(err)
		}
		for j := 0; j < numDims; j++ {
			fmt.Printf("Input %d : dim %d=%d\n", i, j, dims[j])
		}
		typeInfo.ReleaseTypeInfo()
	}

	inputTensorSize := 224 * 224 * 3
	//inputTensorValues := [150528]float64{}
	inputTensorValues := make([]float64, inputTensorSize)
	//outputNodeNames := []string{
	//	"softmaxout_1",
	//}

	for i := 0; i < inputTensorSize; i++ {
		inputTensorValues[i] = float64(i) / float64(inputTensorSize+1)
	}

	memoryInfo, err := ort.NewCPUMemoryInfo(ort.AllocatorTypeArena, ort.MemTypeDefault)
	if err != nil {
		panic(err)
	}

	inData := floatsToBytes(inputTensorValues)
	value, err := ort.NewTensorWithDataAsValue(memoryInfo, inData, tensorInfo)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", value)

	isTensor, err := value.IsTensor()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Is Tensor?: %t\n", isTensor)
	memoryInfo.ReleaseMemoryInfo()

	session.ReleaseSession()
	opts.ReleaseSessionOptions()
	env.ReleaseEnvironment()
}

func floatsToBytes(floats []float64) []byte {
	bytes := make([]byte, len(floats)*8)
	for i := 0; i < len(floats); i++ {
		start := i * 8
		end := start + 8
		binary.LittleEndian.PutUint64(bytes[start:end], math.Float64bits(floats[i]))
	}
	return bytes
}
