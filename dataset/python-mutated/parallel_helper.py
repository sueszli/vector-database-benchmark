import os
from ..framework import Parameter
__parallel_ctx__clz__ = None

def _is_data_parallel_mode():
    if False:
        print('Hello World!')
    global __parallel_ctx__clz__
    return __parallel_ctx__clz__ is not None and int(os.getenv('PADDLE_TRAINERS_NUM', '1')) > 1

def _is_parallel_ctx_initialized():
    if False:
        for i in range(10):
            print('nop')
    global __parallel_ctx__clz__
    return __parallel_ctx__clz__ is not None

def _set_parallel_ctx(ccl_parallel_context):
    if False:
        while True:
            i = 10
    global __parallel_ctx__clz__
    assert __parallel_ctx__clz__ is None, 'ParallelContext can only be initialized once.'
    __parallel_ctx__clz__ = ccl_parallel_context

def _init_parallel_ctx():
    if False:
        print('Hello World!')
    global __parallel_ctx__clz__
    assert __parallel_ctx__clz__ is not None, 'ParallelContext should be initialized.'
    __parallel_ctx__clz__.init()

def _broadcast_parameters(parameters):
    if False:
        return 10
    from ..distributed import broadcast
    for param in parameters:
        if param.is_distributed:
            continue
        if isinstance(param, Parameter) and param.trainable:
            broadcast(param, 0, sync_op=True)