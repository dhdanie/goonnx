package main

/*
#cgo LDFLAGS: -L/usr/local/lib/onnx -lonnxruntime
*/
import "C"
import (
	"fmt"
	"github.com/dhdanie/goonnx/ort"
	"github.com/disintegration/imaging"
	"log"
	"math"
	"os"
	"time"
)

func main() {
	defer timeTrack(time.Now(), "main")

	//rgbs := preprocessImage("images/kitten.jpg")
	rgbs := preprocessImage("images/dog.jpg")
	//rgbs := preprocessImage("images/car.jpg")

	outputs := classifyResNet(rgbs)
	for _, output := range outputs {
		scores := NewScoresFromResults(output)
		scores = Softmax(scores)
		sorted := QuickSort(scores)
		labels, _ := LoadLabels("models/imagenet1000_clsidx_to_labels.txt")
		for i := 0; i < 5; i++ {
			if labels != nil {
				label := labels[sorted[len(sorted)-(i+1)].ClassIndex()]
				fmt.Printf("%f: %s\n", sorted[len(sorted)-(i+1)].Score(), label)
			} else {
				fmt.Printf("%s\n", sorted[len(sorted)-(i+1)])
			}
		}
	}
}

func timeTrack(start time.Time, name string) {
	elapsed := time.Since(start)
	log.Printf("%s took %s", name, elapsed)
}

func errorAndExit(err error) {
	_, _ = fmt.Fprintf(os.Stderr, "Error: %s\n", err.Error())
	os.Exit(1)
}

func Transpose(rgbs []float32) []float32 {
	defer timeTrack(time.Now(), "Transpose")

	out := make([]float32, len(rgbs))
	channelLength := len(rgbs) / 3
	for i := 0; i < channelLength; i++ {
		out[i] = rgbs[i*3]
		out[i+channelLength] = rgbs[i*3+1]
		out[i+channelLength*2] = rgbs[i*3+2]
	}
	return out
}

func preprocessImage(imageFile string) []float32 {
	defer timeTrack(time.Now(), "preprocessImage")

	src, err := imaging.Open(imageFile)
	if err != nil {
		errorAndExit(err)
	}

	rgbs := make([]float32, 224*224*3)

	result := imaging.Resize(src, 256, 256, imaging.Lanczos)
	result = imaging.CropAnchor(result, 224, 224, imaging.Center)
	j := 0
	for i := range result.Pix {
		if (i+1)%4 != 0 {
			rgbs[j] = float32(result.Pix[i])
			j++
		}
	}

	rgbs = Transpose(rgbs)
	channelLength := len(rgbs) / 3
	for i := 0; i < channelLength; i++ {
		rgbs[i] = normalize(rgbs[i]/255, 0.485, 0.229)
		rgbs[i+channelLength] = normalize(rgbs[i+channelLength]/255, 0.456, 0.224)
		rgbs[i+channelLength*2] = normalize(rgbs[i+channelLength*2]/255, 0.406, 0.225)
	}
	return rgbs
}

func normalize(in float32, m float32, s float32) float32 {
	return (in - m) / s
}

func Softmax(in []ClassScore) []ClassScore {
	defer timeTrack(time.Now(), "Softmax")

	out := make([]ClassScore, len(in))

	inMax := max(in)
	var sum float32 = 0.0
	for i, val := range in {
		out[i] = &classScore{
			classIndex: val.ClassIndex(),
			score:      float32(math.Exp(float64(val.Score() - inMax.Score()))),
		}
		sum += out[i].Score()
	}

	for i, val := range out {
		out[i] = &classScore{
			classIndex: val.ClassIndex(),
			score:      RoundFloat32(val.Score() / sum),
		}
	}
	return out
}

func RoundFloat32(in float32) float32 {
	f64in := float64(in)
	return float32(math.Round(f64in*10000000) / 10000000)
}

func max(in []ClassScore) ClassScore {
	defer timeTrack(time.Now(), "max")

	var maxVal float32 = 0.0
	maxIndex := -1
	for i, val := range in {
		if val.Score() > maxVal {
			maxVal = val.Score()
			maxIndex = i
		}
	}
	return in[maxIndex]
}

func classifyResNet(rgbVals []float32) [][]float32 {
	defer timeTrack(time.Now(), "classifyResnet")

	env, err := ort.NewEnvironment(ort.LoggingLevelWarning, "abcde")
	if err != nil {
		errorAndExit(err)
	}
	defer env.ReleaseEnvironment()

	opts := ort.NewSessionOptions()
	opts.SetIntraOpNumThreads(1)
	opts.SetSessionGraphOptimizationLevel(ort.GraphOptLevelEnableBasic)

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
	outs, err := session.Run(ort.NewRunOptions(), inputValues)
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

type ClassScore interface {
	ClassIndex() int
	Score() float32
	Equals(other ClassScore) bool
}

type classScore struct {
	classIndex int
	score      float32
}

func (s *classScore) ClassIndex() int {
	return s.classIndex
}

func (s *classScore) Score() float32 {
	return s.score
}

func (s *classScore) Equals(other ClassScore) bool {
	if s.score == other.Score() && s.classIndex == other.ClassIndex() {
		return true
	}
	return false
}

func (s classScore) String() string {
	return fmt.Sprintf("Class: %d, Score %f", s.classIndex, s.score)
}

func NewScoresFromResults(results []float32) []ClassScore {
	var scores []ClassScore
	for i, result := range results {
		scores = append(scores, &classScore{
			classIndex: i,
			score:      result,
		})
	}
	return scores
}

func QuickSort(scores []ClassScore) []ClassScore {
	var sorted []ClassScore
	sorted = append(sorted, scores...)

	qsort(sorted, 0, len(sorted)-1)

	return sorted
}

func qsort(in []ClassScore, low int, high int) {
	if low < high {
		pi := partition(in, low, high)
		qsort(in, low, pi-1)
		qsort(in, pi+1, high)
	}
}

func partition(in []ClassScore, low int, high int) int {
	pivot := in[high]
	i := low - 1
	for j := low; j <= high-1; j++ {
		if in[j].Score() < pivot.Score() {
			i++
			swap(in, i, j)
		}
	}
	swap(in, i+1, high)
	return i + 1
}

func swap(in []ClassScore, index1 int, index2 int) {
	i1val := in[index1]
	in[index1] = in[index2]
	in[index2] = i1val
}
