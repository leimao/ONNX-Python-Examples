# ONNX Python Examples


## Introduction

ONNX Python examples showing how to create and modify ONNX models using ONNX Python API.


## Usages

### Build Docker Image

```
$ docker build -f docker/onnx.Dockerfile --no-cache --tag=onnx:0.0.1 .
```

### Run Docker Container

```
$ docker run -it --rm --ipc=host -v $(pwd):/mnt onnx:0.0.1
```

### Run Examples

#### Create Neural Network

Create a dummy convolutional neural network from scratch using ONNX Python API.

```
$ python create_convnet.py
```

#### Run Neural Network

Run the dummy convolutional neural network using ONNX Runtime.

```
$ python run_convnet.py
```

#### Modify Neural Network

Modify the dummy convolutional neural network using ONNX Python API.

```
$ python modify_convnet.py
```


## References

* [Creating and Modifying ONNX Model Using ONNX Python API](https://leimao.github.io/blog/ONNX-Python-API/)
* [ONNX.Helper](https://github.com/onnx/onnx/blob/rel-1.9.0/onnx/helper.py)
* [ONNX.Proto](https://github.com/onnx/onnx/blob/rel-1.9.0/onnx/onnx.proto)
* [Python API Overview](https://github.com/onnx/onnx/blob/rel-1.9.0/docs/PythonAPIOverview.md)

