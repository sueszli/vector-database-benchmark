import numpy as np
import torch
import test_external_source_parallel_utils as utils
from nose_utils import raises

class ExtCallbackTorch(utils.ExtCallback):

    def __call__(self, sample_info):
        if False:
            i = 10
            return i + 15
        return torch.tensor(super().__call__(sample_info))

@raises(RuntimeError, 'Error*starting Python worker threads for*parallel External Source*Cannot fork*CUDA has been initialized**start_py_workers*fork*spawn*')
def test_pytorch_cuda_context():
    if False:
        return 10
    cuda0 = torch.device('cuda:0')
    _ = torch.ones([1, 1], dtype=torch.float32, device=cuda0)
    callback = utils.ExtCallback((4, 5), 10, np.int32)
    pipe = utils.create_pipe(callback, 'cpu', 5, py_num_workers=6, py_start_method='fork', parallel=True)
    pipe.start_py_workers()

def test_pytorch():
    if False:
        i = 10
        return i + 15
    yield from utils.check_spawn_with_callback(ExtCallbackTorch)

class ExtCallbackTorchCuda(utils.ExtCallback):

    def __call__(self, sample_info):
        if False:
            while True:
                i = 10
        return torch.tensor(super().__call__(sample_info), device=torch.device('cuda:0'))

@raises(Exception, 'Exception traceback received from worker thread*TypeError: Unsupported callback return type. GPU tensors*not supported*Got*PyTorch GPU tensor')
def test_pytorch_cuda():
    if False:
        i = 10
        return i + 15
    callback = ExtCallbackTorchCuda((4, 5), 10, np.int32)
    pipe = utils.create_pipe(callback, 'cpu', 5, py_num_workers=6, py_start_method='spawn', parallel=True)
    utils.build_and_run_pipeline(pipe)