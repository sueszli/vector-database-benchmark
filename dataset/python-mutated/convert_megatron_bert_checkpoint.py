import argparse
import os
import re
import zipfile
import torch
from transformers import MegatronBertConfig

def recursive_print(name, val, spaces=0):
    if False:
        while True:
            i = 10
    if name is None:
        msg = None
    else:
        fmt = '.' * max(0, spaces - 2) + '# {:' + str(50 - spaces) + 's}'
        msg = fmt.format(name)
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ':', val.size())
    else:
        print(msg, ':', val)

def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    if False:
        print('Hello World!')
    input_shape = param.size()
    if checkpoint_version == 1.0:
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param

def convert_megatron_checkpoint(args, input_state_dict, config):
    if False:
        print('Hello World!')
    output_state_dict = {}
    ds_args = input_state_dict.get('args', None)
    if ds_args is not None:
        config.tokenizer_type = ds_args.tokenizer_type
        config.vocab_size = ds_args.padded_vocab_size
        config.max_position_embeddings = ds_args.max_position_embeddings
        config.hidden_size = ds_args.hidden_size
        config.num_hidden_layers = ds_args.num_layers
        config.num_attention_heads = ds_args.num_attention_heads
        config.intermediate_size = ds_args.ffn_hidden_size if 'ffn_hidden_size' in ds_args else 4 * ds_args.hidden_size
    heads = config.num_attention_heads
    hidden_size_per_head = config.hidden_size // heads
    if 'checkpoint_version' in input_state_dict.keys():
        checkpoint_version = input_state_dict['checkpoint_version']
    else:
        checkpoint_version = 0.0
    model = input_state_dict['model']
    lm = model['language_model']
    embeddings = lm['embedding']
    word_embeddings = embeddings['word_embeddings']['weight']
    word_embeddings = word_embeddings[:config.vocab_size, :]
    output_state_dict['bert.embeddings.word_embeddings.weight'] = word_embeddings
    pos_embeddings = embeddings['position_embeddings']['weight']
    assert pos_embeddings.size(0) == config.max_position_embeddings and pos_embeddings.size(1) == config.hidden_size
    output_state_dict['bert.embeddings.position_embeddings.weight'] = pos_embeddings
    tokentype_embeddings = embeddings['tokentype_embeddings']['weight']
    output_state_dict['bert.embeddings.token_type_embeddings.weight'] = tokentype_embeddings
    transformer = lm['transformer'] if 'transformer' in lm.keys() else lm['encoder']
    layer_re = re.compile('layers\\.(\\d+)\\.([a-z0-9_.]+)\\.([a-z]+)')
    megatron_to_transformers = {'attention.dense': '.attention.output.dense.', 'self_attention.dense': '.attention.output.dense.', 'mlp.dense_h_to_4h': '.intermediate.dense.', 'mlp.dense_4h_to_h': '.output.dense.'}
    attention_qkv_weight = None
    for (key, val) in transformer.items():
        m = layer_re.match(key)
        if m is None:
            break
        layer_idx = int(m.group(1))
        op_name = m.group(2)
        weight_or_bias = m.group(3)
        layer_name = f'bert.encoder.layer.{layer_idx}'
        if op_name.endswith('layernorm'):
            ln_name = 'attention.ln' if op_name.startswith('input') else 'ln'
            output_state_dict[layer_name + '.' + ln_name + '.' + weight_or_bias] = val
        elif (op_name == 'attention.query_key_value' or op_name == 'self_attention.query_key_value') and weight_or_bias == 'weight':
            assert attention_qkv_weight is None, ''
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            attention_qkv_weight = out_val
        elif (op_name == 'attention.query_key_value' or op_name == 'self_attention.query_key_value') and weight_or_bias == 'bias':
            assert attention_qkv_weight is not None, ''
            q = attention_qkv_weight[0 * config.hidden_size:1 * config.hidden_size, :]
            k = attention_qkv_weight[1 * config.hidden_size:2 * config.hidden_size, :]
            v = attention_qkv_weight[2 * config.hidden_size:3 * config.hidden_size, :]
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            q_bias = out_val[0 * config.hidden_size:1 * config.hidden_size]
            k_bias = out_val[1 * config.hidden_size:2 * config.hidden_size]
            v_bias = out_val[2 * config.hidden_size:3 * config.hidden_size]
            output_state_dict[f'{layer_name}.attention.self.query.weight'] = q
            output_state_dict[f'{layer_name}.attention.self.query.bias'] = q_bias
            output_state_dict[f'{layer_name}.attention.self.key.weight'] = k
            output_state_dict[f'{layer_name}.attention.self.key.bias'] = k_bias
            output_state_dict[f'{layer_name}.attention.self.value.weight'] = v
            output_state_dict[f'{layer_name}.attention.self.value.bias'] = v_bias
            attention_qkv_weight = None
        elif weight_or_bias in ['weight', 'bias']:
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + weight_or_bias] = val
    output_state_dict['bert.encoder.ln.weight'] = transformer['final_layernorm.weight']
    output_state_dict['bert.encoder.ln.bias'] = transformer['final_layernorm.bias']
    pooler = lm['pooler']
    output_state_dict['bert.pooler.dense.weight'] = pooler['dense.weight']
    output_state_dict['bert.pooler.dense.bias'] = pooler['dense.bias']
    lm_head = model['lm_head']
    output_state_dict['cls.predictions.transform.dense.weight'] = lm_head['dense.weight']
    output_state_dict['cls.predictions.transform.dense.bias'] = lm_head['dense.bias']
    output_state_dict['cls.predictions.transform.LayerNorm.weight'] = lm_head['layernorm.weight']
    output_state_dict['cls.predictions.transform.LayerNorm.bias'] = lm_head['layernorm.bias']
    output_state_dict['cls.predictions.decoder.weight'] = word_embeddings
    output_state_dict['cls.predictions.bias'] = lm_head['bias']
    binary_head = model['binary_head']
    output_state_dict['cls.seq_relationship.weight'] = binary_head['weight']
    output_state_dict['cls.seq_relationship.bias'] = binary_head['bias']
    return output_state_dict

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--print-checkpoint-structure', action='store_true')
    parser.add_argument('path_to_checkpoint', type=str, help='Path to the ZIP file containing the checkpoint')
    parser.add_argument('--config_file', default='', type=str, help='An optional config json file describing the pre-trained model.')
    args = parser.parse_args()
    basename = os.path.dirname(args.path_to_checkpoint)
    print(f'Extracting PyTorch state dictionary from "{args.path_to_checkpoint}"')
    if args.path_to_checkpoint.endswith('.zip'):
        with zipfile.ZipFile(args.path_to_checkpoint, 'r') as checkpoint:
            with checkpoint.open('release/mp_rank_00/model_optim_rng.pt') as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location='cpu')
    else:
        input_state_dict = torch.load(args.path_to_checkpoint, map_location='cpu')
    if args.config_file == '':
        config = MegatronBertConfig()
        config.vocab_size = input_state_dict['model']['lm_head']['bias'].numel()
    else:
        config = MegatronBertConfig.from_json_file(args.config_file)
    print('Converting')
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)
    print('Saving config')
    config.save_pretrained(basename)
    output_checkpoint_file = os.path.join(basename, 'pytorch_model.bin')
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)
if __name__ == '__main__':
    main()