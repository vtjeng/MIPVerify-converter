#!/usr/bin/env julia

using MIPVerify
using MAT

"""
A basic helper function to test the accuracy of a neural net.
"""
function test_accuracy(network::NeuralNet, dataset::MIPVerify.NamedTrainTestDataset)
    # we're evaluating the test accuracy over the full dataset, but you can always test on a smaller
    # subset of values for larger networks (forward propagation through `NeuralNet`s is still relatively slow)
    println(
        "Fraction of $(dataset.name) test set correct: $(frac_correct(network, dataset.test, length(dataset.test.labels)))"
    )
end

"""
For each new architecture you're working with, you have to re-construct the structure of the neural network. (Sorry!)
Here's what you need to do.

  1. Determine the names of the parameters. A straightforward way to do so is to inspect the model in Netron. https://netron.app/
     See https://github.com/onnx/onnx/issues/1425#issuecomment-636180016 for an example.
  2. Manually re-construct the network using the network primitives here: 
     https://vtjeng.com/MIPVerify.jl/stable/net_components/overview/.
     The names of these primitives largely mirror names from Pytorch, but the convention for the
     layers is not consistent with Pytorch.
       - The `Linear` (a.k.a. fully connected) layer's matrix is tranposed w.r.t the Gemm operator
         https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
       - The `Convolution` layer's convention matches that of Tensorflow.

     You can use these helpers: https://vtjeng.com/MIPVerify.jl/stable/utils/import_weights/ to 
     make working with imported weights more convenient. 
     Note that these helpers expect the parameter names to be of the form LAYER_NAME/KERNEL_NAME
     LAYER_NAME/BIAS_NAME.
"""

mnist = read_datasets("mnist")

## 1. mnist_sample_1
# "mnist_sample_1.jl" contains the function get_mnist_sample_1_network, which re-constructs the structure of the neural network.
include("mnist_sample_1.jl")
model_path = joinpath("networks", "mnist_sample_1.mat")
param_dict = matread(model_path)
mnist_sample_1 = get_mnist_sample_1_network(param_dict, name="mnist_sample_1")
# We make sure that we've imported this network correctly, by validating that the test accuracy is
# as expected.
# For `mnist_sample_1`, the expected fraction correct is 0.9782.
test_accuracy(mnist_sample_1, mnist)

## 2. mnist-net_256x4
# We show results for a different network.
include("mnist-net_256x4.jl")
model_path = joinpath("networks", "mnist-net_256x4.mat")
param_dict = matread(model_path)
mnist_net_256x4 = get_mnist_net_256x4_network(param_dict, name="mnist-net_256x4")
# For `mnist-net_256x4`, the expected fraction correct is 0.9764.
test_accuracy(mnist_net_256x4, mnist)
