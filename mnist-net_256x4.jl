function get_mnist_net_256x4_network(param_dict::Dict; name::String="")
    # For the sample network `mnist-net_256x4.onnx`, the parameter names are
    # layers.0.weight, layers.0.bias, layers.2.weight, layers.2.bias, ...
    
    # `get_matrix_params` expects parameters names to be of the form LAYER_NAME/weight and LAYER_NAME/bias
    p = Dict(replace(name, "." => "/") => value for (name, value) in param_dict)
    for (name, value) in p
        if occursin("weight", name)
            # the convention for the matrix in the standard Gemm layer is the opposite to
            # MIPVerify's `Linear` layer, so we have to transpose the input matrix.
            p[name] = transpose(value)
        end
    end

    dense_1 = get_matrix_params(p, "layers/0", (784, 256))
    dense_2 = get_matrix_params(p, "layers/2", (256, 256))
    dense_3 = get_matrix_params(p, "layers/4", (256, 256))
    dense_4 = get_matrix_params(p, "layers/6", (256, 256))
    dense_5 = get_matrix_params(p, "layers/8", (256, 10))

    # Also, see https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/01_importing_your_own_neural_net.ipynb#Composing-the-network] for an explanation.
    n1 = Sequential([
        # our input is in a 4-dimensional tensor, so we have to flatten the input to begin with.
        Flatten(4),
        dense_1,
        # for optimal solve performance, set the first layer to use only `interval_arithmetic` tightening.
        ReLU(interval_arithmetic),
        dense_2,
        ReLU(),
        dense_3,
        ReLU(),
        dense_4,
        ReLU(),
        dense_5,
    ], name)
end
