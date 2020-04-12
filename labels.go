package main

import (
	"encoding/json"
	"io/ioutil"
)

func LoadLabels(classFile string) (map[int]string, error) {
	data, err := ioutil.ReadFile(classFile)
	if err != nil {
		return nil, err
	}
	var result map[int]string
	err = json.Unmarshal(data, &result)
	if err != nil {
		return nil, err
	}
	return result, nil
}
