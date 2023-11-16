from chainer.backends import cuda
from chainer import function_hook

class CUDAProfileHook(function_hook.FunctionHook):
    name = 'CUDAProfileHook'

    def __init__(self):
        if False:
            while True:
                i = 10
        cuda.check_cuda_available()
        if not cuda.cupy.cuda.nvtx_enabled:
            raise RuntimeError('nvtx is required for CUDAProfileHook')

    def forward_preprocess(self, function, in_data):
        if False:
            for i in range(10):
                print('nop')
        cuda.cupy.cuda.nvtx.RangePush(function.label + '.forward')

    def forward_postprocess(self, function, in_data):
        if False:
            return 10
        cuda.cupy.cuda.nvtx.RangePop()

    def backward_preprocess(self, function, in_data, out_grad):
        if False:
            i = 10
            return i + 15
        cuda.cupy.cuda.nvtx.RangePush(function.label + '.backward')

    def backward_postprocess(self, function, in_data, out_grad):
        if False:
            for i in range(10):
                print('nop')
        cuda.cupy.cuda.nvtx.RangePop()