from paddle import _C_ops
from paddle.framework import LayerHelper, in_dynamic_mode

def masked_multihead_attention(x, cache_kv=None, bias=None, src_mask=None, cum_offsets=None, sequence_lengths=None, rotary_tensor=None, beam_cache_offset=None, qkv_out_scale=None, out_shift=None, out_smooth=None, seq_len=1, rotary_emb_dims=0, use_neox_rotary_style=False, compute_dtype='default', out_scale=-1, quant_round_type=1, quant_max_bound=127.0, quant_min_bound=-127.0):
    if False:
        return 10
    '\n    Masked Multi-head attention for text summarization.\n    This is a fusion operator to compute masked multihead attention in transformer model architecture.\n    This operator only supports running on GPU.\n\n    Args:\n        x (Tensor): The input tensor could be 2-D tensor. Its shape is [batch_size, 3 * num_head * head_dim].\n        cache_kvs (list(Tensor)|tuple(Tensor)): The cache structure tensors for the generation model. Its shape is [2, batch_size, num_head, max_seq_len, head_dim].\n        bias (Tensor, optional): The bias tensor. Its shape is [3, num_head, head_dim].\n        src_mask (Tensor, optional): The src_mask tensor. Its shape is [batch_size, 1, 1, sequence_length].\n        sequence_lengths (Tensor, optional): The sequence_lengths tensor, used to index input. Its shape is [batch_size, 1].\n        rotary_tensor (Tensor, optional): The rotary_tensor tensor. The dtype must be float. Its shape is [batch_size, 1, 1, sequence_length, head_dim].\n        beam_cache_offset (Tensor, optional): The beam_cache_offset tensor. Its shape is [batch_size, beam_size, max_seq_len + max_dec_len].\n        qkv_out_scale (Tensor, optional): The qkv_out_scale tensor, used in quant. Its shape is [3, num_head, head_dim].\n        out_shift (Tensor, optional): The out_shift tensor, used in quant.\n        out_smooth (Tensor, optional): The out_smooth tensor, used in quant.\n        seq_len (int, optional): The seq_len, used to get input length. Default 1.\n        rotary_emb_dims (int, optional): The rotary_emb_dims. Default 1.\n        use_neox_rotary_style (bool, optional): A flag indicating whether neox_rotary_style is needed or not. Default False.\n        compute_dtype (string): A compute dtype, used to represent the input data type.\n        out_scale (float, optional): The out_scale, used in quant.\n        quant_round_type (int, optional): The quant_round_type, used in quant. Default 1.\n        quant_max_bound (float, optional): The quant_max_bound, used in quant. Default 127.0.\n        quant_min_bound (float, optional): The quant_min_bound, used in quant. Default -127.0.\n\n    Returns:\n        Tensor|tuple: If "beam_cache_offset_out" is not none, return the\n        tuple (output, cache_kvs_out, beam_cache_offset_out), which output is the output of\n        masked_multihead_attention layers, cache_kvs_out is inplace with input `cache_kvs`.\n        If "beam_cache_offset_out" is none, return the tuple (output, cache_kvs_out).\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> import paddle\n            >>> import paddle.incubate.nn.functional as F\n            >>> paddle.device.set_device(\'gpu\')\n\n            >>> # input: [batch_size, 3 * num_head * dim_head]\n            >>> x = paddle.rand(shape=(2, 3 * 32 * 128), dtype="float32")\n\n            >>> # src_mask: [batch_size, 1, 1, sequence_length]\n            >>> src_mask = paddle.rand(shape=(2, 1, 1, 10), dtype="float32")\n\n            >>> # cache_kv: [2, batch_size, num_head, max_seq_len, dim_head]\n            >>> cache_kv = paddle.rand(shape=(2, 2, 32, 64, 128), dtype="float32")\n\n            >>> output = F.masked_multihead_attention(\n            ...     x, src_mask=src_mask, cache_kv=cache_kv)\n\n    '
    if in_dynamic_mode():
        return _C_ops.masked_multihead_attention_(x, cache_kv, bias, src_mask, cum_offsets, sequence_lengths, rotary_tensor, beam_cache_offset, qkv_out_scale, out_shift, out_smooth, seq_len, rotary_emb_dims, use_neox_rotary_style, compute_dtype, out_scale, quant_round_type, quant_max_bound, quant_min_bound)
    helper = LayerHelper('masked_multihead_attention', **locals())
    if x.dtype == 'int32':
        if compute_dtype == 'bf16':
            dtype = 'uint16'
        elif compute_dtype == 'fp16':
            dtype = 'float16'
        elif compute_dtype == 'fp32':
            dtype = 'float32'
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    inputs = {}
    inputs['x'] = x
    inputs['cache_kv'] = cache_kv
    if bias is not None:
        inputs['bias'] = bias
    if src_mask is not None:
        inputs['src_mask'] = src_mask
    if cum_offsets is not None:
        inputs['cum_offsets'] = cum_offsets
    if sequence_lengths is not None:
        inputs['sequence_lengths'] = sequence_lengths
    if rotary_tensor is not None:
        inputs['rotary_tensor'] = rotary_tensor
    beam_cache_offset_flag = False
    if beam_cache_offset is not None:
        inputs['beam_cache_offset'] = beam_cache_offset
        beam_cache_offset_flag = True
    else:
        beam_cache_offset = helper.create_variable_for_type_inference(dtype='int')
    if qkv_out_scale is not None:
        inputs['qkv_out_scale'] = qkv_out_scale
    if out_shift is not None:
        inputs['out_shift'] = out_shift
    if out_smooth is not None:
        inputs['out_smooth'] = out_smooth
    outputs = {'out': out, 'cache_kv_out': cache_kv, 'beam_cache_offset_out': beam_cache_offset}
    helper.append_op(type='masked_multihead_attention', inputs=inputs, outputs=outputs, attrs={'seq_len': seq_len, 'rotary_emb_dims': rotary_emb_dims, 'use_neox_rotary_style': use_neox_rotary_style, 'compute_dtype': compute_dtype, 'out_scale': out_scale, 'quant_round_type': quant_round_type, 'quant_max_bound': quant_max_bound, 'quant_min_bound': quant_min_bound})
    return (out, cache_kv, beam_cache_offset) if beam_cache_offset_flag is not None else (out, cache_kv)