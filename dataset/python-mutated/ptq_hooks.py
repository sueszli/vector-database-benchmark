def quant_forward_post_hook(layer, inputs, outputs):
    if False:
        while True:
            i = 10
    '\n    The forward_post_hook for PTQ.\n    '
    assert hasattr(layer, '_quant_config'), 'The layer should have _quant_config attr'
    qc = layer._quant_config
    if qc.enable_in_act_quantizer:
        qc.in_act_quantizer.sample_data(layer, inputs)
    qc.out_act_quantizer.sample_data(layer, (outputs,))