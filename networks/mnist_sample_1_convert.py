#!/usr/bin/env python3

"""
This script has been verified to work with 

keras2onnx==1.7.0
tensorflow==2.3.1

With tensorflow==2.4.1, you will encounter https://github.com/onnx/keras-onnx/issues/651.
"""
import os

from tensorflow import keras
import onnxmltools

SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Download `resources/mnist/mnist-net.h5` from VENUS package, and place it at this path.
model = keras.models.load_model(os.path.join(SCRIPT_DIRECTORY, "mnist_sample_1.h5"))
onnx_model = onnxmltools.convert_keras(model)
onnxmltools.utils.save_model(
    onnx_model, os.path.join(SCRIPT_DIRECTORY, "mnist_sample_1.onnx")
)

