from functools import partial
from . import functions
from . import rpc_async
import torch
from .constants import UNSET_RPC_TIMEOUT
from torch.futures import Future

def _local_invoke(rref, func_name, args, kwargs):
    if False:
        for i in range(10):
            print('nop')
    return getattr(rref.local_value(), func_name)(*args, **kwargs)

@functions.async_execution
def _local_invoke_async_execution(rref, func_name, args, kwargs):
    if False:
        for i in range(10):
            print('nop')
    return getattr(rref.local_value(), func_name)(*args, **kwargs)

def _invoke_rpc(rref, rpc_api, func_name, timeout, *args, **kwargs):
    if False:
        print('Hello World!')

    def _rref_type_cont(rref_fut):
        if False:
            i = 10
            return i + 15
        rref_type = rref_fut.value()
        _invoke_func = _local_invoke
        bypass_type = issubclass(rref_type, torch.jit.ScriptModule) or issubclass(rref_type, torch._C.ScriptModule)
        if not bypass_type:
            func = getattr(rref_type, func_name)
            if hasattr(func, '_wrapped_async_rpc_function'):
                _invoke_func = _local_invoke_async_execution
        return rpc_api(rref.owner(), _invoke_func, args=(rref, func_name, args, kwargs), timeout=timeout)
    rref_fut = rref._get_type(timeout=timeout, blocking=False)
    if rpc_api != rpc_async:
        rref_fut.wait()
        return _rref_type_cont(rref_fut)
    else:
        result: Future = Future()

        def _wrap_rref_type_cont(fut):
            if False:
                for i in range(10):
                    print('nop')
            try:
                _rref_type_cont(fut).then(_complete_op)
            except BaseException as ex:
                result.set_exception(ex)

        def _complete_op(fut):
            if False:
                for i in range(10):
                    print('nop')
            try:
                result.set_result(fut.value())
            except BaseException as ex:
                result.set_exception(ex)
        rref_fut.then(_wrap_rref_type_cont)
        return result

class RRefProxy:

    def __init__(self, rref, rpc_api, timeout=UNSET_RPC_TIMEOUT):
        if False:
            for i in range(10):
                print('nop')
        self.rref = rref
        self.rpc_api = rpc_api
        self.rpc_timeout = timeout

    def __getattr__(self, func_name):
        if False:
            while True:
                i = 10
        return partial(_invoke_rpc, self.rref, self.rpc_api, func_name, self.rpc_timeout)