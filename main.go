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

	inputTensorSize := 224 * 224 * 3
	//inputTensorValues := [150528]float64{}
	inputTensorValues := make([]float32, inputTensorSize)
	//outputNodeNames := []string{
	//	"softmaxout_1",
	//}

	for i := 0; i < inputTensorSize; i++ {
		inputTensorValues[i] = float32(i) / float32(inputTensorSize+1)
	}

	memoryInfo, err := ort.NewCPUMemoryInfo(ort.AllocatorTypeArena, ort.MemTypeDefault)
	if err != nil {
		panic(err)
	}

	typeInfo, err := session.GetInputTypeInfo(0)
	if err != nil {
		panic(err)
	}
	tensorInfo, err := typeInfo.ToTensorInfo()
	if err != nil {
		panic(err)
	}

	//inData := floatsToBytes(inputTensorValues)
	//value, err := ort.NewTensorWithDataAsValue(memoryInfo, inData, tensorInfo)
	value, err := ort.NewTensorWithFloatDataAsValue(memoryInfo, inputTensorValues, tensorInfo)
	if err != nil {
		panic(err)
	}

	isTensor, err := value.IsTensor()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Is Tensor?: %t\n", isTensor)
	memoryInfo.ReleaseMemoryInfo()
	inVals, err := value.GetTensorMutableFloatData()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Size of inVals: %d\n", len(inVals))
	fmt.Printf("Slices Equal: %t\n", slicesEqual(inputTensorValues, inVals))
	session.PrintIOInfo()

	inputValues := []ort.Value{
		value,
	}
	_, err = session.Run(ort.NewRunOptions(), inputValues)
	if err != nil {
		panic(err)
	}

	typeInfo.ReleaseTypeInfo()
	session.ReleaseSession()
	opts.ReleaseSessionOptions()
	env.ReleaseEnvironment()
}

func floatsToBytes(floats []float32) []byte {
	bytes := make([]byte, len(floats)*8)
	for i := 0; i < len(floats); i++ {
		start := i * 8
		end := start + 8
		binary.LittleEndian.PutUint32(bytes[start:end], math.Float32bits(floats[i]))
	}
	return bytes
}

func slicesEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i, aVal := range a {
		if aVal != b[i] {
			return false
		}
	}
	return true
}
