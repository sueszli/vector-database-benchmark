from typing import Optional
import warnings

def get_enum(reduction: str) -> int:
    if False:
        i = 10
        return i + 15
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1
        raise ValueError(f'{reduction} is not a valid value for reduction')
    return ret

def legacy_get_string(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool=True) -> str:
    if False:
        for i in range(10):
            print('nop')
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret

def legacy_get_enum(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool=True) -> int:
    if False:
        print('Hello World!')
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))