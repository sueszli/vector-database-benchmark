from caffe2.python import brew

class AttentionType:
    (Regular, Recurrent, Dot, SoftCoverage) = tuple(range(4))

def s(scope, name):
    if False:
        while True:
            i = 10
    return '{}/{}'.format(str(scope), str(name))

def _calc_weighted_context(model, encoder_outputs_transposed, encoder_output_dim, attention_weights_3d, scope):
    if False:
        return 10
    attention_weighted_encoder_context = brew.batch_mat_mul(model, [encoder_outputs_transposed, attention_weights_3d], s(scope, 'attention_weighted_encoder_context'))
    (attention_weighted_encoder_context, _) = model.net.Reshape(attention_weighted_encoder_context, [attention_weighted_encoder_context, s(scope, 'attention_weighted_encoder_context_old_shape')], shape=[1, -1, encoder_output_dim])
    return attention_weighted_encoder_context

def _calc_attention_weights(model, attention_logits_transposed, scope, encoder_lengths=None):
    if False:
        while True:
            i = 10
    if encoder_lengths is not None:
        attention_logits_transposed = model.net.SequenceMask([attention_logits_transposed, encoder_lengths], ['masked_attention_logits'], mode='sequence')
    attention_weights_3d = brew.softmax(model, attention_logits_transposed, s(scope, 'attention_weights_3d'), engine='CUDNN', axis=1)
    return attention_weights_3d

def _calc_attention_logits_from_sum_match(model, decoder_hidden_encoder_outputs_sum, encoder_output_dim, scope):
    if False:
        for i in range(10):
            print('nop')
    decoder_hidden_encoder_outputs_sum = model.net.Tanh(decoder_hidden_encoder_outputs_sum, decoder_hidden_encoder_outputs_sum)
    attention_logits = brew.fc(model, decoder_hidden_encoder_outputs_sum, s(scope, 'attention_logits'), dim_in=encoder_output_dim, dim_out=1, axis=2, freeze_bias=True)
    attention_logits_transposed = brew.transpose(model, attention_logits, s(scope, 'attention_logits_transposed'), axes=[1, 0, 2])
    return attention_logits_transposed

def _apply_fc_weight_for_sum_match(model, input, dim_in, dim_out, scope, name):
    if False:
        i = 10
        return i + 15
    output = brew.fc(model, input, s(scope, name), dim_in=dim_in, dim_out=dim_out, axis=2)
    output = model.net.Squeeze(output, output, dims=[0])
    return output

def apply_recurrent_attention(model, encoder_output_dim, encoder_outputs_transposed, weighted_encoder_outputs, decoder_hidden_state_t, decoder_hidden_state_dim, attention_weighted_encoder_context_t_prev, scope, encoder_lengths=None):
    if False:
        for i in range(10):
            print('nop')
    weighted_prev_attention_context = _apply_fc_weight_for_sum_match(model=model, input=attention_weighted_encoder_context_t_prev, dim_in=encoder_output_dim, dim_out=encoder_output_dim, scope=scope, name='weighted_prev_attention_context')
    weighted_decoder_hidden_state = _apply_fc_weight_for_sum_match(model=model, input=decoder_hidden_state_t, dim_in=decoder_hidden_state_dim, dim_out=encoder_output_dim, scope=scope, name='weighted_decoder_hidden_state')
    decoder_hidden_encoder_outputs_sum_tmp = model.net.Add([weighted_prev_attention_context, weighted_decoder_hidden_state], s(scope, 'decoder_hidden_encoder_outputs_sum_tmp'))
    decoder_hidden_encoder_outputs_sum = model.net.Add([weighted_encoder_outputs, decoder_hidden_encoder_outputs_sum_tmp], s(scope, 'decoder_hidden_encoder_outputs_sum'), broadcast=1)
    attention_logits_transposed = _calc_attention_logits_from_sum_match(model=model, decoder_hidden_encoder_outputs_sum=decoder_hidden_encoder_outputs_sum, encoder_output_dim=encoder_output_dim, scope=scope)
    attention_weights_3d = _calc_attention_weights(model=model, attention_logits_transposed=attention_logits_transposed, scope=scope, encoder_lengths=encoder_lengths)
    attention_weighted_encoder_context = _calc_weighted_context(model=model, encoder_outputs_transposed=encoder_outputs_transposed, encoder_output_dim=encoder_output_dim, attention_weights_3d=attention_weights_3d, scope=scope)
    return (attention_weighted_encoder_context, attention_weights_3d, [decoder_hidden_encoder_outputs_sum])

def apply_regular_attention(model, encoder_output_dim, encoder_outputs_transposed, weighted_encoder_outputs, decoder_hidden_state_t, decoder_hidden_state_dim, scope, encoder_lengths=None):
    if False:
        i = 10
        return i + 15
    weighted_decoder_hidden_state = _apply_fc_weight_for_sum_match(model=model, input=decoder_hidden_state_t, dim_in=decoder_hidden_state_dim, dim_out=encoder_output_dim, scope=scope, name='weighted_decoder_hidden_state')
    decoder_hidden_encoder_outputs_sum = model.net.Add([weighted_encoder_outputs, weighted_decoder_hidden_state], s(scope, 'decoder_hidden_encoder_outputs_sum'), broadcast=1, use_grad_hack=1)
    attention_logits_transposed = _calc_attention_logits_from_sum_match(model=model, decoder_hidden_encoder_outputs_sum=decoder_hidden_encoder_outputs_sum, encoder_output_dim=encoder_output_dim, scope=scope)
    attention_weights_3d = _calc_attention_weights(model=model, attention_logits_transposed=attention_logits_transposed, scope=scope, encoder_lengths=encoder_lengths)
    attention_weighted_encoder_context = _calc_weighted_context(model=model, encoder_outputs_transposed=encoder_outputs_transposed, encoder_output_dim=encoder_output_dim, attention_weights_3d=attention_weights_3d, scope=scope)
    return (attention_weighted_encoder_context, attention_weights_3d, [decoder_hidden_encoder_outputs_sum])

def apply_dot_attention(model, encoder_output_dim, encoder_outputs_transposed, decoder_hidden_state_t, decoder_hidden_state_dim, scope, encoder_lengths=None):
    if False:
        return 10
    if decoder_hidden_state_dim != encoder_output_dim:
        weighted_decoder_hidden_state = brew.fc(model, decoder_hidden_state_t, s(scope, 'weighted_decoder_hidden_state'), dim_in=decoder_hidden_state_dim, dim_out=encoder_output_dim, axis=2)
    else:
        weighted_decoder_hidden_state = decoder_hidden_state_t
    squeezed_weighted_decoder_hidden_state = model.net.Squeeze(weighted_decoder_hidden_state, s(scope, 'squeezed_weighted_decoder_hidden_state'), dims=[0])
    expanddims_squeezed_weighted_decoder_hidden_state = model.net.ExpandDims(squeezed_weighted_decoder_hidden_state, squeezed_weighted_decoder_hidden_state, dims=[2])
    attention_logits_transposed = model.net.BatchMatMul([encoder_outputs_transposed, expanddims_squeezed_weighted_decoder_hidden_state], s(scope, 'attention_logits'), trans_a=1)
    attention_weights_3d = _calc_attention_weights(model=model, attention_logits_transposed=attention_logits_transposed, scope=scope, encoder_lengths=encoder_lengths)
    attention_weighted_encoder_context = _calc_weighted_context(model=model, encoder_outputs_transposed=encoder_outputs_transposed, encoder_output_dim=encoder_output_dim, attention_weights_3d=attention_weights_3d, scope=scope)
    return (attention_weighted_encoder_context, attention_weights_3d, [])

def apply_soft_coverage_attention(model, encoder_output_dim, encoder_outputs_transposed, weighted_encoder_outputs, decoder_hidden_state_t, decoder_hidden_state_dim, scope, encoder_lengths, coverage_t_prev, coverage_weights):
    if False:
        print('Hello World!')
    weighted_decoder_hidden_state = _apply_fc_weight_for_sum_match(model=model, input=decoder_hidden_state_t, dim_in=decoder_hidden_state_dim, dim_out=encoder_output_dim, scope=scope, name='weighted_decoder_hidden_state')
    decoder_hidden_encoder_outputs_sum_tmp = model.net.Add([weighted_encoder_outputs, weighted_decoder_hidden_state], s(scope, 'decoder_hidden_encoder_outputs_sum_tmp'), broadcast=1)
    coverage_t_prev_2d = model.net.Squeeze(coverage_t_prev, s(scope, 'coverage_t_prev_2d'), dims=[0])
    coverage_t_prev_transposed = brew.transpose(model, coverage_t_prev_2d, s(scope, 'coverage_t_prev_transposed'))
    scaled_coverage_weights = model.net.Mul([coverage_weights, coverage_t_prev_transposed], s(scope, 'scaled_coverage_weights'), broadcast=1, axis=0)
    decoder_hidden_encoder_outputs_sum = model.net.Add([decoder_hidden_encoder_outputs_sum_tmp, scaled_coverage_weights], s(scope, 'decoder_hidden_encoder_outputs_sum'))
    attention_logits_transposed = _calc_attention_logits_from_sum_match(model=model, decoder_hidden_encoder_outputs_sum=decoder_hidden_encoder_outputs_sum, encoder_output_dim=encoder_output_dim, scope=scope)
    attention_weights_3d = _calc_attention_weights(model=model, attention_logits_transposed=attention_logits_transposed, scope=scope, encoder_lengths=encoder_lengths)
    attention_weighted_encoder_context = _calc_weighted_context(model=model, encoder_outputs_transposed=encoder_outputs_transposed, encoder_output_dim=encoder_output_dim, attention_weights_3d=attention_weights_3d, scope=scope)
    attention_weights_2d = model.net.Squeeze(attention_weights_3d, s(scope, 'attention_weights_2d'), dims=[2])
    coverage_t = model.net.Add([coverage_t_prev, attention_weights_2d], s(scope, 'coverage_t'), broadcast=1)
    return (attention_weighted_encoder_context, attention_weights_3d, [decoder_hidden_encoder_outputs_sum], coverage_t)