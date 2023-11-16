import os, sys
import numpy as np
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device, gpu
from cntk.ops.tests.ops_test_utils import cntk_device
import pytest
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'Image', 'Classification', 'ConvNet', 'Python'))
from ConvNet_MNIST import convnet_mnist
TOLERANCE_ABSOLUTE = 0.01

def test_convnet_mnist_error(device_id):
    if False:
        for i in range(10):
            print('nop')
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    try_set_default_device(cntk_device(device_id))
    error = convnet_mnist(epoch_size=5000, minibatch_size=32, max_epochs=10)
    expected_error = 0.0226
    assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)