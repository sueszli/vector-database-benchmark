import sys
import pytest
try:
    import tensorflow as tf
    import torch
    from recommenders.utils.gpu_utils import get_cuda_version, get_cudnn_version, get_gpu_info, get_number_gpus
except ImportError:
    pass

@pytest.mark.gpu
def test_get_gpu_info():
    if False:
        i = 10
        return i + 15
    assert len(get_gpu_info()) >= 1

@pytest.mark.gpu
def test_get_number_gpus():
    if False:
        while True:
            i = 10
    assert get_number_gpus() >= 1

@pytest.mark.gpu
@pytest.mark.skip(reason='TODO: Implement this')
def test_clear_memory_all_gpus():
    if False:
        for i in range(10):
            print('nop')
    pass

@pytest.mark.gpu
@pytest.mark.skipif(sys.platform == 'win32', reason='Not implemented on Windows')
def test_get_cuda_version():
    if False:
        while True:
            i = 10
    assert int(get_cuda_version().split('.')[0]) > 9

@pytest.mark.gpu
def test_get_cudnn_version():
    if False:
        return 10
    assert int(get_cudnn_version()[0]) > 7

@pytest.mark.gpu
def test_cudnn_enabled():
    if False:
        for i in range(10):
            print('nop')
    assert torch.backends.cudnn.enabled == True

@pytest.mark.gpu
@pytest.mark.skip(reason='This function in TF is flaky')
def test_tensorflow_gpu():
    if False:
        while True:
            i = 10
    assert len(tf.config.list_physical_devices('GPU')) > 0

@pytest.mark.gpu
@pytest.mark.skip(reason='This function in PyTorch is flaky')
def test_pytorch_gpu():
    if False:
        for i in range(10):
            print('nop')
    assert torch.cuda.is_available()