from typing import List
import torch

def _validate_dtypes(*dtypes):
    if False:
        print('Hello World!')
    for dtype in dtypes:
        assert isinstance(dtype, torch.dtype)
    return dtypes

class _dispatch_dtypes(tuple):

    def __add__(self, other):
        if False:
            return 10
        assert isinstance(other, tuple)
        return _dispatch_dtypes(tuple.__add__(self, other))
_empty_types = _dispatch_dtypes(())

def empty_types():
    if False:
        while True:
            i = 10
    return _empty_types
_floating_types = _dispatch_dtypes((torch.float32, torch.float64))

def floating_types():
    if False:
        i = 10
        return i + 15
    return _floating_types
_floating_types_and_half = _floating_types + (torch.half,)

def floating_types_and_half():
    if False:
        print('Hello World!')
    return _floating_types_and_half

def floating_types_and(*dtypes):
    if False:
        return 10
    return _floating_types + _validate_dtypes(*dtypes)
_floating_and_complex_types = _floating_types + (torch.cfloat, torch.cdouble)

def floating_and_complex_types():
    if False:
        for i in range(10):
            print('nop')
    return _floating_and_complex_types

def floating_and_complex_types_and(*dtypes):
    if False:
        print('Hello World!')
    return _floating_and_complex_types + _validate_dtypes(*dtypes)
_double_types = _dispatch_dtypes((torch.float64, torch.complex128))

def double_types():
    if False:
        return 10
    return _double_types
_integral_types = _dispatch_dtypes((torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64))

def integral_types():
    if False:
        i = 10
        return i + 15
    return _integral_types

def integral_types_and(*dtypes):
    if False:
        print('Hello World!')
    return _integral_types + _validate_dtypes(*dtypes)
_all_types = _floating_types + _integral_types

def all_types():
    if False:
        print('Hello World!')
    return _all_types

def all_types_and(*dtypes):
    if False:
        while True:
            i = 10
    return _all_types + _validate_dtypes(*dtypes)
_complex_types = _dispatch_dtypes((torch.cfloat, torch.cdouble))

def complex_types():
    if False:
        i = 10
        return i + 15
    return _complex_types

def complex_types_and(*dtypes):
    if False:
        for i in range(10):
            print('nop')
    return _complex_types + _validate_dtypes(*dtypes)
_all_types_and_complex = _all_types + _complex_types

def all_types_and_complex():
    if False:
        print('Hello World!')
    return _all_types_and_complex

def all_types_and_complex_and(*dtypes):
    if False:
        return 10
    return _all_types_and_complex + _validate_dtypes(*dtypes)
_all_types_and_half = _all_types + (torch.half,)

def all_types_and_half():
    if False:
        for i in range(10):
            print('nop')
    return _all_types_and_half

def custom_types(*dtypes):
    if False:
        while True:
            i = 10
    'Create a list of arbitrary dtypes'
    return _empty_types + _validate_dtypes(*dtypes)

def get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=True, include_complex=True, include_complex32=False, include_qint=False) -> List[torch.dtype]:
    if False:
        i = 10
        return i + 15
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(include_half=include_half, include_bfloat16=include_bfloat16)
    if include_bool:
        dtypes.append(torch.bool)
    if include_complex:
        dtypes += get_all_complex_dtypes(include_complex32)
    if include_qint:
        dtypes += get_all_qint_dtypes()
    return dtypes

def get_all_math_dtypes(device) -> List[torch.dtype]:
    if False:
        while True:
            i = 10
    return get_all_int_dtypes() + get_all_fp_dtypes(include_half=device.startswith('cuda'), include_bfloat16=False) + get_all_complex_dtypes()

def get_all_complex_dtypes(include_complex32=False) -> List[torch.dtype]:
    if False:
        while True:
            i = 10
    return [torch.complex32, torch.complex64, torch.complex128] if include_complex32 else [torch.complex64, torch.complex128]

def get_all_int_dtypes() -> List[torch.dtype]:
    if False:
        while True:
            i = 10
    return [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]

def get_all_fp_dtypes(include_half=True, include_bfloat16=True) -> List[torch.dtype]:
    if False:
        while True:
            i = 10
    dtypes = [torch.float32, torch.float64]
    if include_half:
        dtypes.append(torch.float16)
    if include_bfloat16:
        dtypes.append(torch.bfloat16)
    return dtypes

def get_all_qint_dtypes() -> List[torch.dtype]:
    if False:
        return 10
    return [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]
float_to_corresponding_complex_type_map = {torch.float16: torch.complex32, torch.float32: torch.complex64, torch.float64: torch.complex128}