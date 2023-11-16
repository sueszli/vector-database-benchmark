import warnings
import paddle
from paddle.base.wrapped_decorator import wrap_decorator
from paddle.framework import in_dynamic_mode

def _inplace_apis_in_dygraph_only_(func):
    if False:
        while True:
            i = 10

    def __impl__(*args, **kwargs):
        if False:
            return 10
        if not in_dynamic_mode():
            origin_api_name = func.__name__[:-1]
            warnings.warn('In static graph mode, {}() is the same as {}() and does not perform inplace operation.'.format(func.__name__, origin_api_name))
            from ..base.dygraph.base import in_to_static_mode
            if in_to_static_mode():
                for arg in args:
                    if hasattr(arg, 'is_view_var') and arg.is_view_var:
                        raise ValueError(f"Sorry about what's happend. In to_static mode, {func.__name__}'s output variable {arg.name} is a viewed Tensor in dygraph. This will result in inconsistent calculation behavior between dynamic and static graphs. You mast find the location of the strided API be called, and call {arg.name} = {arg.name}.assign().")
            origin_func = f'{func.__module__}.{origin_api_name}'
            return eval(origin_func)(*args, **kwargs)
        return func(*args, **kwargs)
    return __impl__
inplace_apis_in_dygraph_only = wrap_decorator(_inplace_apis_in_dygraph_only_)