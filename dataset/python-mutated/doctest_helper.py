import os
import pkg_resources
_gpu_limit = int(os.getenv('CHAINER_TEST_GPU_LIMIT', '-1'))

def skipif(condition):
    if False:
        for i in range(10):
            print('nop')
    if os.environ.get('READTHEDOCS') == 'True':
        return False
    return condition

def skipif_requires_satisfied(*requirements):
    if False:
        i = 10
        return i + 15
    ws = pkg_resources.WorkingSet()
    try:
        ws.require(*requirements)
    except pkg_resources.ResolutionError:
        return False
    return skipif(True)

def skipif_not_enough_cuda_devices(device_count):
    if False:
        print('Hello World!')
    return skipif(0 <= _gpu_limit < device_count)