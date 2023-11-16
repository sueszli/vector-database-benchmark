def _update_transformers_to_freeze_module(state_dict):
    if False:
        i = 10
        return i + 15
    'Updates pre-trained encoders which were saved prior to the addition of FreezeModule.'
    return {k.replace('encoder_obj.transformer.', 'encoder_obj.transformer.module.') if 'encoder_obj.transformer.module' not in k else k: v for (k, v) in state_dict.items()}

def _update_combiner_no_input_features(state_dict):
    if False:
        i = 10
        return i + 15
    'Removed combiner.input_features from state_dict following DeepSpeed integration.'
    return {k: v for (k, v) in state_dict.items() if not k.startswith('combiner.input_features.')}

def _update_combiner_no_device_tensor(state_dict):
    if False:
        i = 10
        return i + 15
    'Removed device_tensor from state_dict following DeepSpeed integration.'
    return {k: v for (k, v) in state_dict.items() if not k.endswith('device_tensor')}

def update_state_dict(state_dict):
    if False:
        for i in range(10):
            print('nop')
    'Checks state_dict on load, updates state dict if needed.'
    state_dict = _update_transformers_to_freeze_module(state_dict)
    state_dict = _update_combiner_no_input_features(state_dict)
    state_dict = _update_combiner_no_device_tensor(state_dict)
    return state_dict