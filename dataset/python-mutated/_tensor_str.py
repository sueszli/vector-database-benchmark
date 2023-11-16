import contextlib
import dataclasses
import math
import textwrap
from typing import Any, Dict, Optional
import torch
from torch import inf

@dataclasses.dataclass
class __PrinterOptions:
    precision: int = 4
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 80
    sci_mode: Optional[bool] = None
PRINT_OPTS = __PrinterOptions()

def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None):
    if False:
        print('Hello World!')
    "Set options for printing. Items shamelessly taken from NumPy\n\n    Args:\n        precision: Number of digits of precision for floating point output\n            (default = 4).\n        threshold: Total number of array elements which trigger summarization\n            rather than full `repr` (default = 1000).\n        edgeitems: Number of array items in summary at beginning and end of\n            each dimension (default = 3).\n        linewidth: The number of characters per line for the purpose of\n            inserting line breaks (default = 80). Thresholded matrices will\n            ignore this parameter.\n        profile: Sane defaults for pretty printing. Can override with any of\n            the above options. (any one of `default`, `short`, `full`)\n        sci_mode: Enable (True) or disable (False) scientific notation. If\n            None (default) is specified, the value is defined by\n            `torch._tensor_str._Formatter`. This value is automatically chosen\n            by the framework.\n\n    Example::\n\n        >>> # Limit the precision of elements\n        >>> torch.set_printoptions(precision=2)\n        >>> torch.tensor([1.12345])\n        tensor([1.12])\n        >>> # Limit the number of elements shown\n        >>> torch.set_printoptions(threshold=5)\n        >>> torch.arange(10)\n        tensor([0, 1, 2, ..., 7, 8, 9])\n        >>> # Restore defaults\n        >>> torch.set_printoptions(profile='default')\n        >>> torch.tensor([1.12345])\n        tensor([1.1235])\n        >>> torch.arange(10)\n        tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])\n\n    "
    if profile is not None:
        if profile == 'default':
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == 'short':
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == 'full':
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = inf
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth
    PRINT_OPTS.sci_mode = sci_mode

def get_printoptions() -> Dict[str, Any]:
    if False:
        print('Hello World!')
    'Gets the current options for printing, as a dictionary that\n    can be passed as ``**kwargs`` to set_printoptions().\n    '
    return dataclasses.asdict(PRINT_OPTS)

@contextlib.contextmanager
def printoptions(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Context manager that temporarily changes the print options.  Accepted\n    arguments are same as :func:`set_printoptions`.'
    old_kwargs = get_printoptions()
    set_printoptions(**kwargs)
    try:
        yield
    finally:
        set_printoptions(**old_kwargs)

def tensor_totype(t):
    if False:
        i = 10
        return i + 15
    dtype = torch.float if t.is_mps else torch.double
    return t.to(dtype=dtype)

class _Formatter:

    def __init__(self, tensor):
        if False:
            for i in range(10):
                print('nop')
        self.floating_dtype = tensor.dtype.is_floating_point
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1
        with torch.no_grad():
            tensor_view = tensor.reshape(-1)
        if not self.floating_dtype:
            for value in tensor_view:
                value_str = f'{value}'
                self.max_width = max(self.max_width, len(value_str))
        else:
            nonzero_finite_vals = torch.masked_select(tensor_view, torch.isfinite(tensor_view) & tensor_view.ne(0))
            if nonzero_finite_vals.numel() == 0:
                return
            nonzero_finite_abs = tensor_totype(nonzero_finite_vals.abs())
            nonzero_finite_min = tensor_totype(nonzero_finite_abs.min())
            nonzero_finite_max = tensor_totype(nonzero_finite_abs.max())
            for value in nonzero_finite_vals:
                if value != torch.ceil(value):
                    self.int_mode = False
                    break
            if self.int_mode:
                if nonzero_finite_max / nonzero_finite_min > 1000.0 or nonzero_finite_max > 100000000.0:
                    self.sci_mode = True
                    for value in nonzero_finite_vals:
                        value_str = f'{{:.{PRINT_OPTS.precision}e}}'.format(value)
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in nonzero_finite_vals:
                        value_str = f'{value:.0f}'
                        self.max_width = max(self.max_width, len(value_str) + 1)
            elif nonzero_finite_max / nonzero_finite_min > 1000.0 or nonzero_finite_max > 100000000.0 or nonzero_finite_min < 0.0001:
                self.sci_mode = True
                for value in nonzero_finite_vals:
                    value_str = f'{{:.{PRINT_OPTS.precision}e}}'.format(value)
                    self.max_width = max(self.max_width, len(value_str))
            else:
                for value in nonzero_finite_vals:
                    value_str = f'{{:.{PRINT_OPTS.precision}f}}'.format(value)
                    self.max_width = max(self.max_width, len(value_str))
        if PRINT_OPTS.sci_mode is not None:
            self.sci_mode = PRINT_OPTS.sci_mode

    def width(self):
        if False:
            return 10
        return self.max_width

    def format(self, value):
        if False:
            return 10
        if self.floating_dtype:
            if self.sci_mode:
                ret = f'{{:{self.max_width}.{PRINT_OPTS.precision}e}}'.format(value)
            elif self.int_mode:
                ret = f'{value:.0f}'
                if not (math.isinf(value) or math.isnan(value)):
                    ret += '.'
            else:
                ret = f'{{:.{PRINT_OPTS.precision}f}}'.format(value)
        else:
            ret = f'{value}'
        return (self.max_width - len(ret)) * ' ' + ret

def _scalar_str(self, formatter1, formatter2=None):
    if False:
        for i in range(10):
            print('nop')
    if formatter2 is not None:
        real_str = _scalar_str(self.real, formatter1)
        imag_str = (_scalar_str(self.imag, formatter2) + 'j').lstrip()
        if imag_str[0] == '+' or imag_str[0] == '-':
            return real_str + imag_str
        else:
            return real_str + '+' + imag_str
    else:
        return formatter1.format(self.item())

def _vector_str(self, indent, summarize, formatter1, formatter2=None):
    if False:
        print('Hello World!')
    element_length = formatter1.width() + 2
    if formatter2 is not None:
        element_length += formatter2.width() + 1
    elements_per_line = max(1, int(math.floor((PRINT_OPTS.linewidth - indent) / element_length)))

    def _val_formatter(val, formatter1=formatter1, formatter2=formatter2):
        if False:
            return 10
        if formatter2 is not None:
            real_str = formatter1.format(val.real)
            imag_str = (formatter2.format(val.imag) + 'j').lstrip()
            if imag_str[0] == '+' or imag_str[0] == '-':
                return real_str + imag_str
            else:
                return real_str + '+' + imag_str
        else:
            return formatter1.format(val)
    if summarize and (not PRINT_OPTS.edgeitems):
        data = ['...']
    elif summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        data = [_val_formatter(val) for val in self[:PRINT_OPTS.edgeitems].tolist()] + [' ...'] + [_val_formatter(val) for val in self[-PRINT_OPTS.edgeitems:].tolist()]
    else:
        data = [_val_formatter(val) for val in self.tolist()]
    data_lines = [data[i:i + elements_per_line] for i in range(0, len(data), elements_per_line)]
    lines = [', '.join(line) for line in data_lines]
    return '[' + (',' + '\n' + ' ' * (indent + 1)).join(lines) + ']'

def _tensor_str_with_formatter(self, indent, summarize, formatter1, formatter2=None):
    if False:
        print('Hello World!')
    dim = self.dim()
    if dim == 0:
        return _scalar_str(self, formatter1, formatter2)
    if dim == 1:
        return _vector_str(self, indent, summarize, formatter1, formatter2)
    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        slices = [_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter1, formatter2) for i in range(0, PRINT_OPTS.edgeitems)] + ['...'] + [_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter1, formatter2) for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))]
    else:
        slices = [_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter1, formatter2) for i in range(0, self.size(0))]
    tensor_str = (',' + '\n' * (dim - 1) + ' ' * (indent + 1)).join(slices)
    return '[' + tensor_str + ']'

def _tensor_str(self, indent):
    if False:
        i = 10
        return i + 15
    if self.numel() == 0:
        return '[]'
    if self.has_names():
        self = self.rename(None)
    summarize = self.numel() > PRINT_OPTS.threshold
    if self._is_zerotensor():
        self = self.clone()
    if self.is_neg():
        self = self.resolve_neg()
    if self.dtype in [torch.float16, torch.bfloat16, torch.float8_e5m2, torch.float8_e5m2fnuz, torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
        self = self.float()
    if self.dtype is torch.complex32:
        self = self.cfloat()
    if self.dtype.is_complex:
        self = self.resolve_conj()
        real_formatter = _Formatter(get_summarized_data(self.real) if summarize else self.real)
        imag_formatter = _Formatter(get_summarized_data(self.imag) if summarize else self.imag)
        return _tensor_str_with_formatter(self, indent, summarize, real_formatter, imag_formatter)
    else:
        formatter = _Formatter(get_summarized_data(self) if summarize else self)
        return _tensor_str_with_formatter(self, indent, summarize, formatter)

def _add_suffixes(tensor_str, suffixes, indent, force_newline):
    if False:
        print('Hello World!')
    tensor_strs = [tensor_str]
    last_line_len = len(tensor_str) - tensor_str.rfind('\n') + 1
    for suffix in suffixes:
        suffix_len = len(suffix)
        if force_newline or last_line_len + suffix_len + 2 > PRINT_OPTS.linewidth:
            tensor_strs.append(',\n' + ' ' * indent + suffix)
            last_line_len = indent + suffix_len
            force_newline = False
        else:
            tensor_strs.append(', ' + suffix)
            last_line_len += suffix_len + 2
    tensor_strs.append(')')
    return ''.join(tensor_strs)

def get_summarized_data(self):
    if False:
        for i in range(10):
            print('nop')
    dim = self.dim()
    if dim == 0:
        return self
    if dim == 1:
        if self.size(0) > 2 * PRINT_OPTS.edgeitems:
            return torch.cat((self[:PRINT_OPTS.edgeitems], self[-PRINT_OPTS.edgeitems:]))
        else:
            return self
    if not PRINT_OPTS.edgeitems:
        return self.new_empty([0] * self.dim())
    elif self.size(0) > 2 * PRINT_OPTS.edgeitems:
        start = [self[i] for i in range(0, PRINT_OPTS.edgeitems)]
        end = [self[i] for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))]
        return torch.stack([get_summarized_data(x) for x in start + end])
    else:
        return torch.stack([get_summarized_data(x) for x in self])

def _str_intern(inp, *, tensor_contents=None):
    if False:
        for i in range(10):
            print('nop')
    if torch._C._functorch.is_functorch_wrapped_tensor(inp):
        return _functorch_wrapper_str_intern(inp, tensor_contents=tensor_contents)
    is_plain_tensor = type(inp) is torch.Tensor or type(inp) is torch.nn.Parameter
    if inp.is_nested:
        prefix = 'nested_tensor('
    elif is_plain_tensor:
        prefix = 'tensor('
    else:
        prefix = f'{type(inp).__name__}('
    indent = len(prefix)
    suffixes = []
    custom_contents_provided = tensor_contents is not None
    if custom_contents_provided:
        tensor_str = tensor_contents
    (self, tangent) = torch.autograd.forward_ad.unpack_dual(inp)
    if self.device.type != torch._C._get_default_device() or (self.device.type == 'cuda' and torch.cuda.current_device() != self.device.index) or self.device.type == 'mps':
        suffixes.append("device='" + str(self.device) + "'")
    if self.device.type in ['xla', 'lazy', 'ipu', 'mtia']:
        self = self.to('cpu')
    _default_complex_dtype = torch.cdouble if torch.get_default_dtype() == torch.double else torch.cfloat
    has_default_dtype = self.dtype in (torch.get_default_dtype(), _default_complex_dtype, torch.int64, torch.bool)
    if self.is_sparse:
        suffixes.append('size=' + str(tuple(self.shape)))
        from torch._subclasses.fake_tensor import FakeTensor
        if not self.is_meta and (not isinstance(self, FakeTensor)):
            suffixes.append('nnz=' + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append('dtype=' + str(self.dtype))
        if not custom_contents_provided:
            indices_prefix = 'indices=tensor('
            indices = self._indices().detach()
            indices_str = _tensor_str(indices, indent + len(indices_prefix))
            if indices.numel() == 0:
                indices_str += ', size=' + str(tuple(indices.shape))
            values_prefix = 'values=tensor('
            values = self._values().detach()
            values_str = _tensor_str(values, indent + len(values_prefix))
            if values.numel() == 0:
                values_str += ', size=' + str(tuple(values.shape))
            tensor_str = indices_prefix + indices_str + '),\n' + ' ' * indent + values_prefix + values_str + ')'
    elif self.layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
        suffixes.append('size=' + str(tuple(self.shape)))
        suffixes.append('nnz=' + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append('dtype=' + str(self.dtype))
        if not custom_contents_provided:
            (compressed_indices_method, plain_indices_method) = {torch.sparse_csr: (torch.Tensor.crow_indices, torch.Tensor.col_indices), torch.sparse_csc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices), torch.sparse_bsr: (torch.Tensor.crow_indices, torch.Tensor.col_indices), torch.sparse_bsc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices)}[self.layout]
            if self.layout in {torch.sparse_csr, torch.sparse_bsr}:
                (cdimname, pdimname) = ('row', 'column')
            else:
                (cdimname, pdimname) = ('column', 'row')
            compressed_indices_prefix = f'c{cdimname[:3]}_indices=tensor('
            compressed_indices = compressed_indices_method(self).detach()
            compressed_indices_str = _tensor_str(compressed_indices, indent + len(compressed_indices_prefix))
            if compressed_indices.numel() == 0:
                compressed_indices_str += ', size=' + str(tuple(compressed_indices.shape))
            plain_indices_prefix = f'{pdimname[:3]}_indices=tensor('
            plain_indices = plain_indices_method(self).detach()
            plain_indices_str = _tensor_str(plain_indices, indent + len(plain_indices_prefix))
            if plain_indices.numel() == 0:
                plain_indices_str += ', size=' + str(tuple(plain_indices.shape))
            values_prefix = 'values=tensor('
            values = self.values().detach()
            values_str = _tensor_str(values, indent + len(values_prefix))
            if values.numel() == 0:
                values_str += ', size=' + str(tuple(values.shape))
            tensor_str = compressed_indices_prefix + compressed_indices_str + '),\n' + ' ' * indent + plain_indices_prefix + plain_indices_str + '),\n' + ' ' * indent + values_prefix + values_str + ')'
    elif self.is_quantized:
        suffixes.append('size=' + str(tuple(self.shape)))
        if not has_default_dtype:
            suffixes.append('dtype=' + str(self.dtype))
        suffixes.append('quantization_scheme=' + str(self.qscheme()))
        if self.qscheme() == torch.per_tensor_affine or self.qscheme() == torch.per_tensor_symmetric:
            suffixes.append('scale=' + str(self.q_scale()))
            suffixes.append('zero_point=' + str(self.q_zero_point()))
        elif self.qscheme() == torch.per_channel_affine or self.qscheme() == torch.per_channel_symmetric or self.qscheme() == torch.per_channel_affine_float_qparams:
            suffixes.append('scale=' + str(self.q_per_channel_scales()))
            suffixes.append('zero_point=' + str(self.q_per_channel_zero_points()))
            suffixes.append('axis=' + str(self.q_per_channel_axis()))
        if not custom_contents_provided:
            tensor_str = _tensor_str(self.dequantize(), indent)
    elif self.is_nested:
        if not custom_contents_provided:

            def indented_str(s, indent):
                if False:
                    while True:
                        i = 10
                return '\n'.join((f'  {line}' for line in s.split('\n')))
            strs = ',\n'.join((indented_str(str(t), indent + 1) for t in torch.ops.aten.unbind.int(self, 0)))
            tensor_str = f'[\n{strs}\n]'
    elif torch._is_functional_tensor(self):
        prefix = '_to_functional_tensor('
        tensor_str = repr(torch._from_functional_tensor(self))
    else:
        from torch._subclasses.fake_tensor import FakeTensor
        if self.is_meta or isinstance(self, FakeTensor):
            suffixes.append('size=' + str(tuple(self.shape)))
            if self.dtype != torch.get_default_dtype():
                suffixes.append('dtype=' + str(self.dtype))
            if not custom_contents_provided:
                tensor_str = '...'
        elif self.numel() == 0 and (not self.is_sparse):
            if self.dim() != 1:
                suffixes.append('size=' + str(tuple(self.shape)))
            if self.dtype != torch.get_default_dtype():
                suffixes.append('dtype=' + str(self.dtype))
            if not custom_contents_provided:
                tensor_str = '[]'
        else:
            if not PRINT_OPTS.edgeitems:
                suffixes.append('size=' + str(tuple(self.shape)))
            if not has_default_dtype:
                suffixes.append('dtype=' + str(self.dtype))
            if not custom_contents_provided:
                if self.layout != torch.strided:
                    tensor_str = _tensor_str(self.to_dense(), indent)
                else:
                    tensor_str = _tensor_str(self, indent)
    if self.layout != torch.strided:
        suffixes.append('layout=' + str(self.layout))
    if inp.grad_fn is not None:
        name = type(inp.grad_fn).__name__
        if name == 'CppFunction':
            name = inp.grad_fn.name().rsplit('::', 1)[-1]
        suffixes.append(f'grad_fn=<{name}>')
    elif inp.requires_grad:
        suffixes.append('requires_grad=True')
    if self.has_names():
        suffixes.append(f'names={self.names}')
    if tangent is not None:
        suffixes.append(f'tangent={tangent}')
    string_repr = _add_suffixes(prefix + tensor_str, suffixes, indent, force_newline=self.is_sparse)
    if isinstance(self, torch.nn.Parameter) and (not is_plain_tensor):
        string_repr = f'Parameter({string_repr})'
    return string_repr

def _functorch_wrapper_str_intern(tensor, *, tensor_contents=None):
    if False:
        i = 10
        return i + 15
    level = torch._C._functorch.maybe_get_level(tensor)
    assert level != -1
    if torch._C._functorch.is_functionaltensor(tensor):
        torch._sync(tensor)
    value = torch._C._functorch.get_unwrapped(tensor)
    value_repr = repr(value)
    indented_value_repr = textwrap.indent(value_repr, ' ' * 4)
    if torch._C._functorch.is_batchedtensor(tensor):
        bdim = torch._C._functorch.maybe_get_bdim(tensor)
        assert bdim != -1
        return f'BatchedTensor(lvl={level}, bdim={bdim}, value=\n{indented_value_repr}\n)'
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        return f'GradTrackingTensor(lvl={level}, value=\n{indented_value_repr}\n)'
    if torch._C._functorch.is_functionaltensor(tensor):
        return f'FunctionalTensor(lvl={level}, value=\\\n{value_repr})'
    raise ValueError("We don't know how to print this, please file us an issue")

def _str(self, *, tensor_contents=None):
    if False:
        print('Hello World!')
    with torch.no_grad(), torch.utils._python_dispatch._disable_current_modes():
        guard = torch._C._DisableFuncTorch()
        return _str_intern(self, tensor_contents=tensor_contents)