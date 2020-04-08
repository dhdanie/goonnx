# Go ONNX

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