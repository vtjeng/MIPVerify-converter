function get_mnist_sample_1_network(param_dict::Dict; name::String="")
    # For the sample network `mnist_sample_1.onnx`, the parameter names are
    # dense_1/kernel:0, dense_1/bias:0, dense_2/kernel:0, dense_2/bias:0, dense_3/kernel:0, dense_3/bias:0
    dense_1 = get_matrix_params(param_dict, "dense_1", (784, 512), matrix_name="kernel:0", bias_name="bias:0")
    dense_2 = get_matrix_params(param_dict, "dense_2", (512, 512), matrix_name="kernel:0", bias_name="bias:0")
    dense_3 = get_matrix_params(param_dict, "dense_3", (512, 10), matrix_name="kernel:0", bias_name="bias:0")

    # Also, see https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/01_importing_your_own_neural_net.ipynb#Composing-the-network] for an explanation.
    n1 = Sequential([
        # our input is in a 4-dimensional tensor, so we have to begin by flattening the input
        Flatten(4),
        dense_1,
        # for optimal solve speed, set the first layer to use only `interval_arithmetic` tightening.
        ReLU(interval_arithmetic),
        dense_2,
        ReLU(),
        dense_3,
    ], name)
end
