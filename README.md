# go-onnx

Go language bindings for ONNX runtime

## About
I'm a fan of Go and have just started digging a bit deeper in to machine learning.  I heard about ONNX runtime and I'm
a fan of standardization, so it seemed like a good place to start.  I realized ONNX runtime didn't have Go language
bindings, and I figured, if I can get that going, it'd probably be a great way to get started on my AI/ML journey.

The initial goal was to replicate the functionality of the C example from the ONNX repository,
[here](https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp).

At this point, the implemented functionality achieves the same result as the example noted above and I've, additionally, 
tested it with ResNet on image classification (example below and in main.go).

The API is incomplete (compared to the functionality available in the C library), at this time.  I may try continue to
build it out, as time permits, but would gladly accept help if anybody else is interested in this sort of thing.

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

You'll also need to download the example ResNet model from [here](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.onnx).

Then, you should be able to run a basic demo application like the following (see main.go for working demo):

```go
package main

/*
#cgo LDFLAGS: -L/usr/local/lib/onnx -lonnxruntime
*/
import "C"
```
...
```go
func classifyResNet(rgbVals []float32) [][]float32 {
	defer timeTrack(time.Now(), "classifyResnet")

	logId := "log0001"

	var myCustomLogger ort.CustomLogger = func(severity ort.LoggingLevel, category string, codeLocation string, message string) {
		fmt.Printf("Custom Logger %d/%s/%s - %s\n", severity, category, codeLocation, message)
	}

	env, err := ort.NewEnvironmentWithCustomLogger(ort.LoggingLevelVerbose, logId, myCustomLogger)
	if err != nil {
		errorAndExit(err)
	}
	defer env.ReleaseEnvironment()

	opts := &ort.SessionOptions{
		IntraOpNumThreads:      1,
		GraphOptimizationLevel: ort.GraphOptLevelEnableBasic,
		SessionLogID:           logId,
		LogVerbosityLevel:      0,
	}

	session, err := ort.NewSession(env, "models/resnet152v2.onnx", opts)
	if err != nil {
		errorAndExit(err)
	}
	defer session.ReleaseSession()

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
	defer memoryInfo.ReleaseMemoryInfo()
	value, err := ort.NewTensorWithFloatDataAsValue(memoryInfo, "data", rgbVals, tensorInfo)
	if err != nil {
		errorAndExit(err)
	}
	inputValues := []ort.Value{
		value,
	}
	outs, err := session.Run(&ort.RunOptions{}, inputValues)
	if err != nil {
		errorAndExit(err)
	}
	outputs := make([][]float32, len(outs))
	for i, out := range outs {
		if out.GetName() != "resnetv27_dense0_fwd" {
			continue
		}
		outFloats, err := out.GetTensorMutableFloatData()
		if err != nil {
			errorAndExit(err)
		}
		outputs[i] = make([]float32, len(outFloats))
		for j := range outFloats {
			outputs[i][j] = outFloats[j]
		}
	}

	return outputs
}
```

## License
[MIT License](LICENSE)