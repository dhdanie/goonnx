package main

import (
	"fmt"
	"github.com/dhdanie/goonnx/ort"
	"os"
)

func errorAndExit(err error) {
	_, _ = fmt.Fprintf(os.Stderr, "Error: %s\n", err.Error())
	os.Exit(1)
}

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
		errorAndExit(err)
	}

	inputTensorSize := 224 * 224 * 3
	inputTensorValues := make([]float32, inputTensorSize)

	for i := 0; i < inputTensorSize; i++ {
		inputTensorValues[i] = float32(i) / float32(inputTensorSize+1)
	}

	typeInfo, err := session.GetInputTypeInfo(0)
	if err != nil {
		errorAndExit(err)
	}
	tensorInfo, err := typeInfo.ToTensorInfo()
	if err != nil {
		errorAndExit(err)
	}
	memoryInfo, err := ort.NewCPUMemoryInfo(ort.AllocatorTypeArena, ort.MemTypeDefault)
	if err != nil {
		errorAndExit(err)
	}
	value, err := ort.NewTensorWithFloatDataAsValue(memoryInfo, inputTensorValues, tensorInfo)
	if err != nil {
		errorAndExit(err)
	}
	memoryInfo.ReleaseMemoryInfo()

	session.PrintIOInfo()
	inputValues := []ort.Value{
		value,
	}
	outs, err := session.Run(ort.NewRunOptions(), inputValues)
	if err != nil {
		errorAndExit(err)
	}
	for _, out := range outs {
		outFloats, err := out.GetTensorMutableFloatData()
		if err != nil {
			errorAndExit(err)
		}
		for i := 0; i < 5; i++ {
			fmt.Printf("Score for class [%d] =  %f\n", i, outFloats[i])
		}
	}

	typeInfo.ReleaseTypeInfo()
	session.ReleaseSession()
	opts.ReleaseSessionOptions()
	env.ReleaseEnvironment()
}
