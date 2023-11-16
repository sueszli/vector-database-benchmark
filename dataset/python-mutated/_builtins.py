import cmath
import math
import warnings
from collections import OrderedDict
from typing import Dict, Optional
import torch
import torch.backends.cudnn as cudnn
from ..nn.modules.utils import _list_with_default, _pair, _quadruple, _single, _triple
_builtin_table: Optional[Dict[int, str]] = None
_modules_containing_builtins = (torch, torch._C._nn, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._sparse, torch._C._special)
_builtin_ops = [(_pair, 'aten::_pair'), (_quadruple, 'aten::_quadruple'), (_single, 'aten::_single'), (_triple, 'aten::_triple'), (_list_with_default, 'aten::list_with_default'), (OrderedDict, 'aten::dict'), (dict, 'aten::dict'), (cudnn.is_acceptable, 'aten::cudnn_is_acceptable'), (math.ceil, 'aten::ceil'), (math.copysign, 'aten::copysign'), (math.erf, 'aten::erf'), (math.erfc, 'aten::erfc'), (math.exp, 'aten::exp'), (math.expm1, 'aten::expm1'), (math.fabs, 'aten::fabs'), (math.floor, 'aten::floor'), (math.gamma, 'aten::gamma'), (math.lgamma, 'aten::lgamma'), (math.log, 'aten::log'), (math.log10, 'aten::log10'), (math.log1p, 'aten::log1p'), (math.pow, 'aten::pow'), (math.sqrt, 'aten::sqrt'), (math.isnan, 'aten::isnan'), (math.asinh, 'aten::asinh'), (math.atanh, 'aten::atanh'), (math.cosh, 'aten::cosh'), (math.sinh, 'aten::sinh'), (math.tanh, 'aten::tanh'), (math.acos, 'aten::acos'), (math.asin, 'aten::asin'), (math.atan, 'aten::atan'), (math.atan2, 'aten::atan2'), (math.cos, 'aten::cos'), (math.sin, 'aten::sin'), (math.tan, 'aten::tan'), (math.asinh, 'aten::asinh'), (math.atanh, 'aten::atanh'), (math.acosh, 'aten::acosh'), (math.fmod, 'aten::fmod'), (math.modf, 'aten::modf'), (math.factorial, 'aten::factorial'), (math.frexp, 'aten::frexp'), (math.isinf, 'aten::isinf'), (math.degrees, 'aten::degrees'), (math.radians, 'aten::radians'), (cmath.isnan, 'aten::isnan'), (cmath.isfinite, 'aten::isfinite'), (cmath.isinf, 'aten::isinf'), (cmath.phase, 'aten::angle'), (cmath.rect, 'aten::polar'), (cmath.log, 'aten::log'), (cmath.log10, 'aten::log10'), (cmath.sqrt, 'aten::sqrt'), (cmath.exp, 'aten::exp'), (cmath.sin, 'aten::sin'), (cmath.tan, 'aten::tan'), (cmath.cos, 'aten::cos'), (cmath.asin, 'aten::asin'), (cmath.acos, 'aten::acos'), (cmath.atan, 'aten::atan'), (cmath.sinh, 'aten::sinh'), (cmath.cosh, 'aten::cosh'), (cmath.tanh, 'aten::tanh'), (cmath.asinh, 'aten::asinh'), (cmath.acosh, 'aten::acosh'), (cmath.atanh, 'aten::atanh'), (math.ldexp, 'aten::ldexp'), (torch._assert, 'aten::_assert'), (torch.autograd.grad, 'aten::grad'), (torch.autograd.backward, 'aten::backward'), (torch._C._infer_size, 'aten::_infer_size'), (torch.nn.functional._no_grad_embedding_renorm_, 'aten::_no_grad_embedding_renorm_'), (torch.nn.functional.assert_int_or_pair, 'aten::_assert_int_or_pair'), (torch.nn.init._no_grad_fill_, 'aten::_no_grad_fill_'), (torch.nn.init._no_grad_normal_, 'aten::_no_grad_normal_'), (torch.nn.init._no_grad_uniform_, 'aten::_no_grad_uniform_'), (torch.nn.init._no_grad_zero_, 'aten::_no_grad_zero_'), (torch._C._get_tracing_state, 'aten::_get_tracing_state'), (torch._C._get_cpu_capability, 'aten::_get_cpu_capability'), (warnings.warn, 'aten::warn'), (torch._VF.stft, 'aten::stft'), (torch._VF.istft, 'aten::istft'), (torch._VF.cdist, 'aten::cdist'), (torch._VF.norm, 'aten::norm'), (torch._VF.unique_dim, 'aten::unique_dim'), (torch._VF.unique_consecutive, 'aten::unique_consecutive'), (torch._VF.nuclear_norm, 'aten::nuclear_norm'), (torch._VF.frobenius_norm, 'aten::frobenius_norm'), (torch._VF.tensordot, 'aten::tensordot')]

def _gen_torch_functional_registered_ops():
    if False:
        return 10
    ops = ['stft', 'istft', 'lu', 'cdist', 'norm', 'unique', 'unique_consecutive', 'tensordot']
    return {getattr(torch.functional, name) for name in ops}
_functional_registered_ops = _gen_torch_functional_registered_ops()

def _is_special_functional_bound_op(fn):
    if False:
        return 10
    return fn in _functional_registered_ops

def _get_builtin_table():
    if False:
        print('Hello World!')
    global _builtin_table
    if _builtin_table is not None:
        return _builtin_table
    _builtin_table = {}

    def register_all(mod):
        if False:
            print('Hello World!')
        for name in dir(mod):
            v = getattr(mod, name)
            if callable(v) and (not _is_special_functional_bound_op(v)) and (v is not torch.no_grad) and (v is not torch.autocast):
                if name == '_segment_reduce':
                    name = name[1:]
                _builtin_ops.append((v, 'aten::' + name))
    for mod in _modules_containing_builtins:
        register_all(mod)
    _builtin_ops.append((math.gcd, 'aten::gcd'))
    _builtin_ops.append((math.isfinite, 'aten::isfinite'))
    _builtin_ops.append((math.remainder, 'aten::mathremainder'))
    import torch.distributed.autograd as dist_autograd
    if dist_autograd.is_available():
        _builtin_ops.append((dist_autograd.get_gradients, 'aten::get_gradients'))
        _builtin_ops.append((dist_autograd.backward, 'aten::dist_backward'))
    for (builtin, aten_op) in _builtin_ops:
        _builtin_table[id(builtin)] = aten_op
    return _builtin_table

def _register_builtin(fn, op):
    if False:
        print('Hello World!')
    _get_builtin_table()[id(fn)] = op

def _find_builtin(fn):
    if False:
        return 10
    return _get_builtin_table().get(id(fn))