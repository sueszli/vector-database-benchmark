import torch
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import TensorLikeType
from torch._prims_common.wrappers import out_wrapper
from torch._refs import _broadcast_shapes
__all__ = ['bfloat16', 'bool', 'byte', 'cdouble', 'cfloat', 'chalf', 'char', 'double', 'float', 'half', 'int', 'long', 'short', 'complex', 'polar']

def _make_conversion_method(name: str, dtype: torch.dtype):
    if False:
        return 10

    def fn(self: TensorLikeType, memory_format: torch.memory_format=torch.preserve_format) -> TensorLikeType:
        if False:
            i = 10
            return i + 15
        return self.to(dtype, memory_format=memory_format)
    fn.__name__ = name
    return fn
bfloat16 = _make_conversion_method('bfloat16', torch.bfloat16)
bool = _make_conversion_method('bool', torch.bool)
byte = _make_conversion_method('byte', torch.uint8)
cdouble = _make_conversion_method('cdouble', torch.cdouble)
cfloat = _make_conversion_method('cfloat', torch.cfloat)
chalf = _make_conversion_method('chalf', torch.complex32)
char = _make_conversion_method('char', torch.int8)
double = _make_conversion_method('double', torch.double)
float = _make_conversion_method('float', torch.float)
half = _make_conversion_method('half', torch.half)
int = _make_conversion_method('int', torch.int)
long = _make_conversion_method('long', torch.long)
short = _make_conversion_method('short', torch.short)

@register_decomposition(torch._ops.ops.aten.complex)
@out_wrapper(exact_dtype=True)
def complex(real: TensorLikeType, imag: TensorLikeType) -> TensorLikeType:
    if False:
        print('Hello World!')
    allowed_dtypes = (torch.float32, torch.float64, torch.float16)
    torch._check(real.dtype in allowed_dtypes and imag.dtype in allowed_dtypes, lambda : f'Expected both inputs to be Half, Float or Double tensors but got {real.dtype} and {imag.dtype}')
    torch._check(real.dtype == imag.dtype, lambda : f'Expected object of scalar type {real.dtype} but got scalar type {imag.dtype} for second argument')
    result_dtype = utils.corresponding_complex_dtype(real.dtype)
    common_shape = _broadcast_shapes(real.shape, imag.shape)
    result = real.new_empty(common_shape, dtype=result_dtype, layout=real.layout, device=real.device)
    result.real = real
    result.imag = imag
    return result

@register_decomposition(torch._ops.ops.aten.polar)
@out_wrapper(exact_dtype=True)
def polar(abs: TensorLikeType, angle: TensorLikeType) -> TensorLikeType:
    if False:
        while True:
            i = 10
    result = torch.complex(abs, angle)
    result.real = abs * torch.cos(angle)
    result.imag = abs * torch.sin(angle)
    return result