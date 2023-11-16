"""Convert SwitchTransformersX checkpoints from the original repository to JAX/FLAX model."""
import argparse
import re
from flax.traverse_util import flatten_dict, unflatten_dict
from t5x import checkpoints
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model
from transformers.utils import logging
logging.set_verbosity_info()
MOE_LAYER_NAME_MAPPING = {'/attention/': '/0/SelfAttention/', '/self_attention/': '/0/SelfAttention/', '/encoder_decoder_attention/': '/1/EncDecAttention/', 'value': 'v', 'query': 'q', 'key': 'k', 'out': 'o', 'pre_self_attention_layer_norm': '0/layer_norm', 'pre_cross_attention_layer_norm': '1/layer_norm', 'pre_attention_layer_norm': '0/layer_norm', 'token_embedder': 'shared', 'encoder_norm': 'final_layer_norm', 'decoder_norm': 'final_layer_norm', 'relpos_bias/rel_embedding': 'block/0/layer/0/SelfAttention/relative_attention_bias/weight', 'router/router_weights/w/': 'router/classifier/', 'roer/roer_weights/w/': 'router/classifier/', 'logits_dense': 'lm_head'}

def rename_keys(s_dict):
    if False:
        i = 10
        return i + 15
    keys = list(s_dict.keys())
    for key in keys:
        layer_to_block_of_layer = '.*/layers_(\\d+)'
        new_key = key
        if re.match(layer_to_block_of_layer, key):
            new_key = re.sub('layers_(\\d+)', 'block/\\1/layer', new_key)
        layer_to_block_of_layer = '(encoder|decoder)\\/'
        if re.match(layer_to_block_of_layer, key):
            groups = re.match(layer_to_block_of_layer, new_key).groups()
            if groups[0] == 'encoder':
                new_key = re.sub('/mlp/', '/1/mlp/', new_key)
                new_key = re.sub('/pre_mlp_layer_norm/', '/1/layer_norm/', new_key)
            elif groups[0] == 'decoder':
                new_key = re.sub('/mlp/', '/2/mlp/', new_key)
                new_key = re.sub('/pre_mlp_layer_norm/', '/2/layer_norm/', new_key)
        for (old_key, temp_key) in MOE_LAYER_NAME_MAPPING.items():
            if old_key in new_key:
                new_key = new_key.replace(old_key, temp_key)
        print(f'{key} -> {new_key}')
        s_dict[new_key] = s_dict.pop(key)
    if 'encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight' in s_dict:
        s_dict['encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight'] = s_dict['encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight'].T
    if 'decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight' in s_dict:
        s_dict['decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight'] = s_dict['decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight'].T
    for key in list(s_dict.keys()):
        if 'expert' in key:
            num_experts = s_dict[key].shape[0]
            expert_weihts = s_dict[key]
            for idx in range(num_experts):
                s_dict[key.replace('expert/', f'experts/expert_{idx}/')] = expert_weihts[idx]
                print(f"{key} -> {key.replace('expert/', f'experts/expert_{idx}/')}")
            s_dict.pop(key)
    return s_dict
GIN_TO_CONFIG_MAPPING = {'NUM_ENCODER_LAYERS': 'num_layers', 'NUM_DECODER_LAYERS': 'num_decoder_layers', 'NUM_HEADS': 'num_heads', 'HEAD_DIM': 'd_kv', 'EMBED_DIM': 'd_model', 'MLP_DIM': 'd_ff', 'NUM_SELECTED_EXPERTS': 'num_selected_experts', 'NUM_ENCODER_SPARSE_LAYERS': 'num_sparse_encoder_layers', 'NUM_DECODER_SPARSE_LAYERS': 'num_sparse_decoder_layers', 'dense.MlpBlock.activations': 'feed_forward_proj'}

def convert_gin_to_config(gin_file, num_experts):
    if False:
        for i in range(10):
            print('nop')
    import regex as re
    with open(gin_file, 'r') as f:
        raw_gin = f.read()
    regex_match = re.findall('(.*) = ([0-9.]*)', raw_gin)
    args = {}
    for (param, value) in regex_match:
        if param in GIN_TO_CONFIG_MAPPING and value != '':
            args[GIN_TO_CONFIG_MAPPING[param]] = float(value) if '.' in value else int(value)
    activation = re.findall("(.*activations) = \\(\\'(.*)\\',\\)", raw_gin)[0]
    args[GIN_TO_CONFIG_MAPPING[activation[0]]] = str(activation[1])
    args['num_experts'] = num_experts
    config = SwitchTransformersConfig(**args)
    return config

def convert_flax_checkpoint_to_pytorch(flax_checkpoint_path, config_file, gin_file=None, pytorch_dump_path='./', num_experts=8):
    if False:
        i = 10
        return i + 15
    print(f'Loading flax weights from : {flax_checkpoint_path}')
    flax_params = checkpoints.load_t5x_checkpoint(flax_checkpoint_path)
    if gin_file is not None:
        config = convert_gin_to_config(gin_file, num_experts)
    else:
        config = SwitchTransformersConfig.from_pretrained(config_file)
    pt_model = SwitchTransformersForConditionalGeneration(config)
    flax_params = flax_params['target']
    flax_params = flatten_dict(flax_params, sep='/')
    flax_params = rename_keys(flax_params)
    flax_params = unflatten_dict(flax_params, sep='/')
    load_flax_weights_in_pytorch_model(pt_model, flax_params)
    print(f'Save PyTorch model to {pytorch_dump_path}')
    pt_model.save_pretrained(pytorch_dump_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--switch_t5x_checkpoint_path', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained SwitchTransformers model. \nThis specifies the model architecture. If not provided, a `gin_file` has to be provided.')
    parser.add_argument('--gin_file', default=None, type=str, required=False, help='Path to the gin config file. If not provided, a `config_file` has to be passed   ')
    parser.add_argument('--config_name', default=None, type=str, required=False, help='Config name of SwitchTransformers model.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output pytorch model.')
    parser.add_argument('--num_experts', default=8, type=int, required=False, help='Number of experts')
    args = parser.parse_args()
    convert_flax_checkpoint_to_pytorch(args.switch_t5x_checkpoint_path, args.config_name, args.gin_file, args.pytorch_dump_folder_path, args.num_experts)