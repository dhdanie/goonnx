# go-onnx

Go language bindings for ONNX runtime

## About
I'm a fan of Go and have just started digging a bit deeper in to machine learning.  I heard about ONNX runtime and I'm
a fan of standardization, so it seemed like a good place to start.  I realized ONNX runtime didn't have Go language
bindings, and I figured, if I can get that going, it'd probably be a great way to get started on my AI/ML journey.

The initial goal was to replicate the functionality of the C example from the ONNX repository,
[here](https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp).

At this point, similar logic as what is performed in the C example has been implemented in this repository.  I
leveraged the ONNX runtime shared library and CGo (which I'm also new at) for the core functionality and created a
pretty basic Go facade for the reference functionality.

I may take this further, but at this point, it's tbd.

## Using this library
**Go-onnx** uses *cgo* and leverages the *onnxruntime* shared library, so to run your program which leverages
**go-onnx**, you'll need to let *cgo* know where that library resides on your local system.  To do so, in your `main.go`
(or wherever), include something like the following snippet:

```go
/*
#cgo LDFLAGS: -L/path/to/onnx/runtime/lib
 */
import "C"
```

The directory specified should contain the `libonnxruntime.so` (named the same).  If your ONNX runtime file is named
something different, you may need to include the additional flag `-l<libname>`.

## Example
For a new application, first get **go-onnx**:

`go get github.com/dhdanie/goonnx`

You'll also need to download the example squeezenet model from [here](https://s3.amazonaws.com/download.onnx/models/opset_9/squeezenet.tar.gz).

Then, you should be able to run a basic demo application like the following:

```go
package main

/*
#cgo LDFLAGS: -L/usr/local/lib/onnx -lonnxruntime
 */
import "C"
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
	env, err := ort.NewEnvironment(ort.LoggingLevelWarning, "demo")
	if err != nil {
		errorAndExit(err)
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		errorAndExit(err)
	}
	err = opts.SetIntraOpNumThreads(1)
	if err != nil {
		errorAndExit(err)
	}
	err = opts.SetSessionGraphOptimizationLevel(ort.GraphOptLevelEnableBasic)
	if err != nil {
		errorAndExit(err)
	}

	session, err := ort.NewSession(env, "/path/to/squeeznet/model.onnx", opts)
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
	typeInfo.ReleaseTypeInfo()

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

	session.ReleaseSession()
	opts.ReleaseSessionOptions()
	env.ReleaseEnvironment()
}
```