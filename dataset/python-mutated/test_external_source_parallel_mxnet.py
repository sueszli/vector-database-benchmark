import mxnet as mx
from nose import with_setup
from nose_utils import raises
from test_pool_utils import setup_function
from test_external_source_parallel_utils import ExtCallback, check_spawn_with_callback, create_pipe, build_and_run_pipeline
import numpy as np

class ExtCallbackMX(ExtCallback):

    def __call__(self, sample_info):
        if False:
            for i in range(10):
                print('nop')
        a = super().__call__(sample_info)
        return mx.nd.array(a, dtype=a.dtype)

def test_mxnet():
    if False:
        for i in range(10):
            print('nop')
    yield from check_spawn_with_callback(ExtCallbackMX)

class ExtCallbackMXCuda(ExtCallback):

    def __call__(self, sample_info):
        if False:
            for i in range(10):
                print('nop')
        a = super().__call__(sample_info)
        return mx.nd.array(a, dtype=a.dtype, ctx=mx.gpu(0))

@raises(Exception, 'Exception traceback received from worker thread*TypeError: Unsupported callback return type. GPU tensors*not supported*Got*MXNet GPU tensor.')
@with_setup(setup_function)
def test_mxnet_cuda():
    if False:
        return 10
    callback = ExtCallbackMXCuda((4, 5), 10, np.int32)
    pipe = create_pipe(callback, 'cpu', 5, py_num_workers=6, py_start_method='spawn', parallel=True)
    build_and_run_pipeline(pipe)