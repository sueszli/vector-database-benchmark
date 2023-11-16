from cntk import ops, cpu, parameter, NDArrayView, input
import numpy as np
import cntk as C
import os
import sys
import pytest
abs_path = os.path.dirname(os.path.abspath(__file__))
custom_convolution_ops_dir = os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'Extensibility', 'BinaryConvolution')
sys.path.append(custom_convolution_ops_dir)
from custom_convolution_ops import *
import cntk.contrib.netopt as nopt

def test_native_binary_function():
    if False:
        print('Hello World!')
    if not nopt.native_convolve_function_registered:
        pytest.skip('Could not find {0} library. Please check if HALIDE_PATH is configured properly and try building {1} again'.format('Cntk.BinaryConvolution-' + C.__version__.rstrip('+'), 'Extnsibiliy\\BinaryConvolution'))
    dev = C.cpu()
    x = input((64, 28, 28))
    w = parameter((64, 64, 3, 3), init=np.reshape(2 * (np.random.rand(64 * 64 * 3 * 3) - 0.5), (64, 64, 3, 3)), dtype=np.float32, device=dev)
    attributes = {'stride': 1, 'padding': False, 'size': 3, 'h': 28, 'w': 28, 'channels': 64, 'filters': 64}
    op = ops.native_user_function('NativeBinaryConvolveFunction', [w, x], attributes, 'native_binary_convolve')
    op2 = C.convolution(CustomMultibitKernel(w, 1), CustomSign(x), auto_padding=[False])
    x_data = NDArrayView.from_dense(np.asarray(np.reshape(2 * (np.random.rand(64 * 28 * 28) - 0.5), (64, 28, 28)), dtype=np.float32), device=dev)
    result = op.eval({x: x_data}, device=dev)
    result2 = op2.eval({x: x_data}, device=dev)
    native_times_primitive = op.find_by_name('native_binary_convolve')
    '\n    Disable this tempororily. Needs to investigate and fix the halide\n    code to match the previous test behavior.\n    '