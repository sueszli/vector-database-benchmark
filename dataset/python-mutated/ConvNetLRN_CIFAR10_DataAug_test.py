import numpy as np
import os
import sys
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device
import pytest
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'Image', 'Classification', 'ConvNet', 'Python'))
from prepare_test_data import prepare_CIFAR10_data
from ConvNetLRN_CIFAR10_DataAug import convnetlrn_cifar10_dataaug, create_reader

def test_cifar_convnet_error(device_id):
    if False:
        i = 10
        return i + 15
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    try_set_default_device(cntk_device(device_id))
    base_path = prepare_CIFAR10_data()
    os.chdir(base_path)
    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    set_computation_network_trace_level(1)
    set_fixed_random_seed(1)
    reader_train = create_reader(os.path.join(base_path, 'train_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), True)
    reader_test = create_reader(os.path.join(base_path, 'test_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), False)
    test_error = convnetlrn_cifar10_dataaug(reader_train, reader_test, epoch_size=256, max_epochs=1)