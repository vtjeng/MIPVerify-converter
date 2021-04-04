# MIPVerify-converter

This repository provides

## Common Issues

```sh
AssertionError: Number of output channels in matrix, 784, does not match number of output channels in bias, 256
```

```sh
MethodError: no method matching get_matrix_params(::Dict{Any,Any}, ::String, ::Tuple{Int64,Int64})
Closest candidates are:
  get_matrix_params(::Dict{String,V} where V, ::String, ::Tuple{Int64,Int64}; matrix_name, bias_name) at /home/vtjeng/.julia/packages/MIPVerify/aGOKf/src/utils/import_weights.jl:23
```

You may have created a parameter dictionary without [explicitly specify]

```julia
p = Dict{String, Array{Float32}}()
...
```

## Sample Networks

| Name                  | Test Accuracy | Source                                                                                                                                       |
| --------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `mnist_sample_1.onnx` | 0.9782        | Courtesy [Matthias K ̈onig](https://www.universiteitleiden.nl/en/staffmembers/matthias-konig#tab-1)                                           |
| `mnist-net_256x4.onnx | 0.9764        | [VNN-Comp 2020 Benchmark](https://github.com/verivital/vnn-comp/tree/5d146cb1c0179a97fc75a3521883d6765142f092/2020/PWL/benchmark/mnist/oval) |
