import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import filter_nodes, fwd_only, joint_fwd_bwd, register_replacement
log = logging.getLogger(__name__)
aten = torch.ops.aten

def _sfdp_pattern_1(query, key, value, inv_scale):
    if False:
        return 10
    return torch.matmul(query, key.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(value)

def _sfdp_replacement_1(query, key, value, inv_scale):
    if False:
        print('Hello World!')
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=None, dropout_p=0.0, is_causal=False, scale=1.0 / inv_scale)

def _sfdp_pattern_2(query, key, value, scale_factor):
    if False:
        for i in range(10):
            print('nop')
    return torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1).matmul(value)

def _sfdp_replacement_2(query, key, value, scale_factor):
    if False:
        print('Hello World!')
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale_factor)

def _sfdp_pattern_3(query, key, value, inv_scale_factor, dropout_p):
    if False:
        return 10
    return torch.nn.functional.dropout(torch.matmul(query, key.transpose(-2, -1)).div(inv_scale_factor).softmax(dim=-1), p=dropout_p).matmul(value)

def _sfdp_replacement_3(query, key, value, inv_scale_factor, dropout_p):
    if False:
        print('Hello World!')
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=None, dropout_p=dropout_p, is_causal=False, scale=1.0 / inv_scale_factor)

def _sfdp_pattern_4(query, key, value, scale_factor, dropout_p):
    if False:
        return 10
    return torch.nn.functional.dropout(torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1), p=dropout_p).matmul(value)

def _sfdp_replacement_4(query, key, value, scale_factor, dropout_p):
    if False:
        while True:
            i = 10
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=None, dropout_p=dropout_p, is_causal=False, scale=scale_factor)

def _sfdp_pattern_5(query, key, value, attn_mask):
    if False:
        print('Hello World!')
    attn_weight = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask, dim=-1)
    return attn_weight @ value

def _sfdp_replacement_5(query, key, value, attn_mask):
    if False:
        i = 10
        return i + 15
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=attn_mask.to(dtype=query.dtype), dropout_p=0.0, is_causal=False)

def _sfdp_pattern_6(query, key, value, attn_mask, dropout_p):
    if False:
        while True:
            i = 10
    attn_weight = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight @ value

def _sfdp_replacement_6(query, key, value, attn_mask, dropout_p):
    if False:
        i = 10
        return i + 15
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=attn_mask.to(dtype=query.dtype), dropout_p=dropout_p, is_causal=False)

def _sfdp_pattern_7(query, key, value, dropout_p):
    if False:
        print('Hello World!')
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v

def _sfdp_replacement_7(query, key, value, dropout_p):
    if False:
        while True:
            i = 10
    counters['inductor']['fuse_attention'] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return aten.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False)

def _sfdp_pattern_8(query, key, value):
    if False:
        i = 10
        return i + 15
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v

def _sfdp_replacement_8(query, key, value):
    if False:
        print('Hello World!')
    counters['inductor']['fuse_attention'] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return aten.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

def _sfdp_pattern_9(query, key, value, dropout_p):
    if False:
        print('Hello World!')
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    q = q / math.sqrt(q.size(-1))
    div = q @ k.transpose(-2, -1)
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v

def _sfdp_replacement_9(query, key, value, dropout_p):
    if False:
        print('Hello World!')
    counters['inductor']['fuse_attention'] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return aten.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False)

def _sfdp_pattern_10(query, key, value):
    if False:
        while True:
            i = 10
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    q = q / math.sqrt(q.size(-1))
    div = q @ k.transpose(-2, -1)
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v

def _sfdp_replacement_10(query, key, value):
    if False:
        print('Hello World!')
    counters['inductor']['fuse_attention'] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return aten.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

def _sfdp_pattern_11(query, key, value, inv_scale):
    if False:
        return 10
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return torch.matmul(q, k.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(v)

def _sfdp_replacement_11(query, key, value, inv_scale):
    if False:
        while True:
            i = 10
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=None, dropout_p=0.0, is_causal=False, scale=1.0 / inv_scale)

def _sfdp_pattern_12(query, key, value, inv_scale_factor, dropout_p):
    if False:
        while True:
            i = 10
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return torch.nn.functional.dropout(torch.matmul(q, k.transpose(-2, -1)).div(inv_scale_factor).softmax(dim=-1), p=dropout_p).matmul(v)

def _sfdp_replacement_12(query, key, value, inv_scale_factor, dropout_p):
    if False:
        for i in range(10):
            print('nop')
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=None, dropout_p=dropout_p, is_causal=False, scale=1.0 / inv_scale_factor)

def _sfdp_pattern_13(query, key, value, dropout_p):
    if False:
        for i in range(10):
            print('nop')
    attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
    attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
    return torch.bmm(attn_weight, value)

def _sfdp_replacement_13(query, key, value, dropout_p):
    if False:
        for i in range(10):
            print('nop')
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), dropout_p=dropout_p, scale=1.0).squeeze(0)

def _sfdp_params_check(match):
    if False:
        for i in range(10):
            print('nop')
    assert all((k in match.kwargs for k in ('query', 'key', 'value')))
    query = match.kwargs['query'].meta['val']
    key = match.kwargs['key'].meta['val']
    value = match.kwargs['value'].meta['val']
    if not query.dtype == key.dtype == value.dtype or not query.device == key.device == value.device:
        return False
    add_mask_node = filter_nodes(match.nodes, aten.add.Tensor)
    if len(add_mask_node) > 0:
        attn_mask_node = add_mask_node[0].args[1]
        if not hasattr(attn_mask_node, 'meta'):
            return False
        attn_mask = attn_mask_node.meta['val']
        if not isinstance(attn_mask, torch.Tensor) or not (attn_mask.dtype == query.dtype or attn_mask.dtype == torch.bool) or query.device != attn_mask.device:
            return False
    return True

def _sfdp_scale_factor_check(scale_factor_op):
    if False:
        while True:
            i = 10

    def fn(match):
        if False:
            i = 10
            return i + 15
        scale_factor_node = filter_nodes(match.nodes, scale_factor_op)[0]
        scale_factor = scale_factor_node.args[1]
        if not isinstance(scale_factor, (float, int)):
            return False
        return _sfdp_params_check(match)
    return fn

def partialize_and_update_signature(func, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Equivalent to functools.partial but also updates the signature on returned function\n    '
    original_sig = inspect.signature(func)
    parameters = original_sig.parameters
    new_parameters = {key: value for (key, value) in parameters.items() if key not in kwargs}
    new_sig = inspect.Signature(parameters=list(new_parameters.values()))
    partial_func = functools.partial(func, **kwargs)

    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return partial_func(*args, **kwargs)
    wrapper.__signature__ = new_sig
    wrapper.__name__ = func.__name__
    return wrapper

def _get_sfdp_patterns():
    if False:
        print('Hello World!')
    from .joint_graph import patterns
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    g_inp = functools.partial(torch.empty, (2, 4, 8, 16), device=device, requires_grad=True)
    b_inp = functools.partial(torch.empty, (1, 1, 8, 8), device=device)
    c_inp = functools.partial(torch.tensor, 2.0, device=device)
    d = {'dropout_p': 0.113377}
    g_3d_inp = functools.partial(torch.empty, (1024, 128, 128), device=device, requires_grad=True)
    for dtype in [torch.float, torch.half]:
        g = functools.partial(g_inp, dtype=dtype)
        b = functools.partial(b_inp, dtype=dtype)
        c = functools.partial(c_inp, dtype=dtype)
        g_3d = functools.partial(g_3d_inp, dtype=dtype)
        for (pattern, replacement, args, workaround, extra_check) in [(_sfdp_pattern_1, _sfdp_replacement_1, [g(), g(), g(), c()], {}, _sfdp_scale_factor_check(aten.div.Tensor)), (_sfdp_pattern_2, _sfdp_replacement_2, [g(), g(), g(), c()], {}, _sfdp_scale_factor_check(aten.mul.Tensor)), (_sfdp_pattern_3, _sfdp_replacement_3, [g(), g(), g(), c()], d, _sfdp_scale_factor_check(aten.div.Tensor)), (_sfdp_pattern_4, _sfdp_replacement_4, [g(), g(), g(), c()], d, _sfdp_scale_factor_check(aten.mul.Tensor)), (_sfdp_pattern_5, _sfdp_replacement_5, [g(), g(), g(), b()], {}, _sfdp_params_check), (_sfdp_pattern_6, _sfdp_replacement_6, [g(), g(), g(), b()], d, _sfdp_params_check), (_sfdp_pattern_7, _sfdp_replacement_7, [g(), g(), g()], d, _sfdp_params_check), (_sfdp_pattern_8, _sfdp_replacement_8, [g(), g(), g()], {}, _sfdp_params_check), (_sfdp_pattern_9, _sfdp_replacement_9, [g(), g(), g()], d, _sfdp_params_check), (_sfdp_pattern_10, _sfdp_replacement_10, [g(), g(), g()], {}, _sfdp_params_check), (_sfdp_pattern_11, _sfdp_replacement_11, [g(), g(), g(), c()], {}, _sfdp_scale_factor_check(aten.div.Tensor)), (_sfdp_pattern_12, _sfdp_replacement_12, [g(), g(), g(), c()], d, _sfdp_scale_factor_check(aten.div.Tensor)), (_sfdp_pattern_13, _sfdp_replacement_13, [g_3d(), g_3d(), g_3d()], d, _sfdp_params_check)]:
            assert isinstance(workaround, dict)
            name = pattern.__name__
            training_name = f'{name}_training' if dtype == torch.float else f'{name}_training_half'
            yield (training_name, {'search_fn': pattern, 'replace_fn': replacement, 'example_inputs': args, 'trace_fn': joint_fwd_bwd, 'pass_dicts': patterns, 'extra_check': extra_check, 'scalar_workaround': workaround})
            if workaround:
                assert len(workaround) == 1 and 'dropout_p' in workaround
                pattern = partialize_and_update_signature(pattern, dropout_p=0.0)
                replacement = partialize_and_update_signature(replacement, dropout_p=0.0)
                workaround = {}
            inference_name = f'{name}_inference' if dtype == torch.float else f'{name}_inference_half'
            yield (inference_name, {'search_fn': pattern, 'replace_fn': replacement, 'example_inputs': args, 'trace_fn': fwd_only, 'pass_dicts': patterns, 'extra_check': extra_check, 'scalar_workaround': workaround})

@functools.lru_cache(None)
def _sfdp_init():
    if False:
        while True:
            i = 10
    from .serialized_patterns.central_index import get_serialized_pattern
    for (key, register_replacement_kwargs) in _get_sfdp_patterns():
        search_fn_pattern = get_serialized_pattern(key)
        register_replacement(**register_replacement_kwargs, search_fn_pattern=search_fn_pattern)