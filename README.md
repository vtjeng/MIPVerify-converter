# MIPVerify-converter

This repository provides a starting point for reading in models for verification via the [`MIPVerify.jl`](https://github.com/vtjeng/MIPVerify.jl/) tool. It is updated for the [v0.3.1 release](https://github.com/vtjeng/MIPVerify.jl/releases/tag/v0.3.1).

## Quick Start

1. **`?` -> `.onnx`**:
   - Models are expected in the [`onnx` format](https://onnx.ai/). Most common frameworks / tools [can easily export to the `onnx` format](https://github.com/onnx/tutorials#converting-to-onnx-format).
2. **`.onnx` -> `.mat`**:
   - `convert.py` extracts the model weights and biases, saving them to a`.mat` file.
     - You will need to install the Python packages specified in [`REQUIREMENTS.txt`](REQUIREMENTS.txt). This can either be done system-wide or [using a virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments).
     - The `networks` folder provides examples of two models in `.onnx` format, with the corresponding `.mat` files produced by `convert.py`. (Note that the `.mat` binary files produced differ at the byte level from run to run, but contain the same data.)

```sh
# after installing REQUIREMENTS
$ ./convert.py -i networks/mnist_sample_1.onnx -o networks/mnist_sample_1.mat
```

3. **`.mat` -> `MIPVerify`-compatible model specification**:
   - Specify the structure of the model using `MIPVerify` primitives in Julia. See [`check.jl`](check.jl) for instructions, and [`mnist_sample_1.jl`](mnist_sample_1.jl) and [`mnist-net_256x4.jl](mnist-net_256x4.jl) for examples.
   - Import the model weights (again, see [`check.jl`](check.jl)) and validate that the model has the expected accuracy.

> :warning: You should need to write Julia code once for each architecture.

## Common Issues

We list some common issues (in **bold**), along with possible causes.

### Model Accuracy

- **Model has reasonable test accuracy but different from expected value.**
  - Your model expects input in a range other than `[0, 1]` (e.g. `[0, 255]` is common for MNIST and CIFAR datasets)
  - Your model expects input [with normalized mean and standard deviation](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Normalize), but the normalization operation has not been serialized in `onnx`.
  - Your model expects input images that are transposed vis-a-vis our dataset.
  - You have [transposed](https://pytorch.org/docs/stable/generated/torch.transpose.html) the two dimensions processing the images along the height and the width respectively.
- **Model has test accuracy no better than random guessing.**
  - _If a convolution layer is present_: you have reshaped the tensor for the weight of the convolution layer incorrectly.

### Miscellaneous Errors

> ```sh
> AssertionError: Number of output channels in matrix, 784, does not match number of output channels in bias, 256
> ```

This class of error message often occurs when the weights of the Linear layer are transposed relative to the convention expected by `MIPVerify`. Transpose the weights when importing them (see [`mnist-net_256x4.jl`](mnist-net_256x4.jl) for an example of how.)

## Sample Networks

The reference test accuracies of the sample MNIST classifier networks are provided below.

| Name                   | Test Accuracy | Source                                                                                                                                                                                                                                                                                         |
| ---------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mnist_sample_1.onnx`  | 0.9782        | Network found at `resources/mnist/mnist-net.h5` in `venus-1.0.1`, retrieved 2021-04-08 from the [VAS Group](https://vas.doc.ic.ac.uk/software/neural/) website. Converted to `.onnx` format courtesy [Matthias König](https://www.universiteitleiden.nl/en/staffmembers/matthias-konig#tab-1). |
| `mnist-net_256x4.onnx` | 0.9764        | [VNN-Comp 2020 Benchmark](https://github.com/verivital/vnn-comp/tree/5d146cb1c0179a97fc75a3521883d6765142f092/2020/PWL/benchmark/mnist/oval)                                                                                                                                                   |
