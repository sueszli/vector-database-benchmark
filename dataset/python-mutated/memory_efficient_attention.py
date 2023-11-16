import paddle
from paddle import _C_ops
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode
from .attn_bias import BlockDiagonalCausalMask, BlockDiagonalCausalWithOffsetPaddedKeysMask, BlockDiagonalMask, LowerTriangularMask, LowerTriangularMaskWithTensorBias
SUPPORTED_ATTN_BIAS_TYPES = {type(None), paddle.Tensor, LowerTriangularMask, LowerTriangularMaskWithTensorBias, BlockDiagonalMask, BlockDiagonalCausalMask, BlockDiagonalCausalWithOffsetPaddedKeysMask}

def _get_seqlen_info(attn_bias):
    if False:
        print('Hello World!')
    if isinstance(attn_bias, (BlockDiagonalMask, BlockDiagonalCausalWithOffsetPaddedKeysMask)):
        return (attn_bias.k_seqinfo.seqstart, attn_bias.q_seqinfo.seqstart, attn_bias.q_seqinfo.max_seqlen, attn_bias.k_seqinfo.max_seqlen)
    else:
        return (None, None, -1, -1)

def _get_tensor_bias(attn_bias):
    if False:
        print('Hello World!')
    if isinstance(attn_bias, paddle.Tensor):
        return attn_bias
    elif isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
        return attn_bias._bias
    else:
        return None

def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, training=True):
    if False:
        return 10
    assert type(attn_bias) in SUPPORTED_ATTN_BIAS_TYPES
    causal = isinstance(attn_bias, (LowerTriangularMask, BlockDiagonalCausalMask, BlockDiagonalCausalWithOffsetPaddedKeysMask))
    (seqstart_k, seqstart_q, max_seqlen_q, max_seqlen_k) = _get_seqlen_info(attn_bias)
    causal_diagonal = attn_bias.causal_diagonal if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask) else None
    seqlen_k = attn_bias.k_seqinfo.seqlen if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask) else None
    if scale is None:
        scale = -1.0
    bias = _get_tensor_bias(attn_bias)
    is_test = not training
    if in_dynamic_mode():
        (output, logsumexp, seed_and_offset) = _C_ops.memory_efficient_attention(query, key, value, bias, seqstart_q, seqstart_k, causal_diagonal, seqlen_k, max_seqlen_q, max_seqlen_k, causal, p, scale, is_test)
        return output
    helper = LayerHelper('memory_efficient_attention', **locals())
    output = helper.create_variable_for_type_inference(dtype=query.dtype)
    logsumexp = helper.create_variable_for_type_inference(dtype='float')
    seed_and_offset = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(type='memory_efficient_attention', inputs={'query': query, 'key': key, 'value': value, 'bias': bias, 'cu_seqlens_q': seqstart_q, 'cu_seqlens_k': seqstart_k, 'causal_diagonal': causal_diagonal, 'seqlen_k': seqlen_k}, attrs={'max_seqlen_q': max_seqlen_q, 'max_seqlen_k': max_seqlen_k, 'causal': causal, 'dropout_p': p, 'scale': scale, 'is_test': is_test}, outputs={'output': output, 'logsumexp': logsumexp, 'seed_and_offset': seed_and_offset})
    return output