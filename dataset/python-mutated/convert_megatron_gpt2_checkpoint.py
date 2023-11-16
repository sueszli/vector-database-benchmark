import argparse
import os
import re
import zipfile
import torch
from transformers import AutoTokenizer, GPT2Config

def recursive_print(name, val, spaces=0):
    if False:
        return 10
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
        for i in range(10):
            print('nop')
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
        i = 10
        return i + 15
    output_state_dict = {}
    ds_args = input_state_dict.get('args', None)
    if ds_args is not None:
        config.vocab_size = ds_args.padded_vocab_size
        config.n_positions = ds_args.max_position_embeddings
        config.n_embd = ds_args.hidden_size
        config.n_layer = ds_args.num_layers
        config.n_head = ds_args.num_attention_heads
        config.n_inner = ds_args.ffn_hidden_size
    heads = config.n_head
    hidden_size_per_head = config.n_embd // config.n_head
    if 'checkpoint_version' in input_state_dict.keys():
        checkpoint_version = input_state_dict['checkpoint_version']
    else:
        checkpoint_version = 0.0
    model = input_state_dict['model']
    lm = model['language_model']
    embeddings = lm['embedding']
    word_embeddings = embeddings['word_embeddings']['weight']
    word_embeddings = word_embeddings[:config.vocab_size, :]
    output_state_dict['transformer.wte.weight'] = word_embeddings
    pos_embeddings = embeddings['position_embeddings']['weight']
    n_positions = pos_embeddings.size(0)
    if n_positions != config.n_positions:
        raise ValueError(f"pos_embeddings.max_sequence_length={n_positions} and config.n_positions={config.n_positions} don't match")
    output_state_dict['transformer.wpe.weight'] = pos_embeddings
    transformer = lm['transformer'] if 'transformer' in lm.keys() else lm['encoder']
    layer_re = re.compile('layers\\.(\\d+)\\.([a-z0-9_.]+)\\.([a-z]+)')
    megatron_to_transformers = {'attention.dense': '.attn.c_proj.', 'self_attention.dense': '.attn.c_proj.', 'mlp.dense_h_to_4h': '.mlp.c_fc.', 'mlp.dense_4h_to_h': '.mlp.c_proj.'}
    for (key, val) in transformer.items():
        m = layer_re.match(key)
        if m is None:
            break
        layer_idx = int(m.group(1))
        op_name = m.group(2)
        weight_or_bias = m.group(3)
        layer_name = f'transformer.h.{layer_idx}'
        if op_name.endswith('layernorm'):
            ln_name = 'ln_1' if op_name.startswith('input') else 'ln_2'
            output_state_dict[layer_name + '.' + ln_name + '.' + weight_or_bias] = val
        elif (op_name == 'attention.query_key_value' or op_name == 'self_attention.query_key_value') and weight_or_bias == 'weight':
            causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.float16)).view(1, 1, n_positions, n_positions)
            output_state_dict[layer_name + '.attn.bias'] = causal_mask
            masked_bias = torch.tensor(-10000.0, dtype=torch.float16)
            output_state_dict[layer_name + '.attn.masked_bias'] = masked_bias
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            out_val = out_val.transpose(0, 1).contiguous()
            output_state_dict[layer_name + '.attn.c_attn.weight'] = out_val
        elif (op_name == 'attention.query_key_value' or op_name == 'self_attention.query_key_value') and weight_or_bias == 'bias':
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            output_state_dict[layer_name + '.attn.c_attn.bias'] = out_val
        elif weight_or_bias == 'weight':
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + 'weight'] = val.transpose(0, 1)
        elif weight_or_bias == 'bias':
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + 'bias'] = val
    assert config.n_layer == layer_idx + 1
    output_state_dict['transformer.ln_f.weight'] = transformer['final_layernorm.weight']
    output_state_dict['transformer.ln_f.bias'] = transformer['final_layernorm.bias']
    output_state_dict['lm_head.weight'] = word_embeddings
    return output_state_dict

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--print-checkpoint-structure', action='store_true')
    parser.add_argument('path_to_checkpoint', type=str, help='Path to the checkpoint file (.zip archive or direct .pt file)')
    parser.add_argument('--config_file', default='', type=str, help='An optional config json file describing the pre-trained model.')
    args = parser.parse_args()
    basename = os.path.dirname(args.path_to_checkpoint)
    print(f'Extracting PyTorch state dictionary from {args.path_to_checkpoint}')
    if args.path_to_checkpoint.endswith('.zip'):
        with zipfile.ZipFile(args.path_to_checkpoint, 'r') as checkpoint:
            with checkpoint.open('release/mp_rank_00/model_optim_rng.pt') as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location='cpu')
    else:
        input_state_dict = torch.load(args.path_to_checkpoint, map_location='cpu')
    ds_args = input_state_dict.get('args', None)
    if args.config_file == '':
        if ds_args is not None:
            if ds_args.bias_gelu_fusion:
                activation_function = 'gelu_fast'
            elif ds_args.openai_gelu:
                activation_function = 'gelu_new'
            else:
                activation_function = 'gelu'
        else:
            activation_function = 'gelu_new'
        config = GPT2Config(vocab_size=50257, n_positions=1024, n_embd=1024, n_layer=24, n_head=16, n_inner=4096, activation_function=activation_function, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02, summary_type='cls_index', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, scale_attn_weights=True, use_cache=True, bos_token_id=50256, eos_token_id=50256)
    else:
        config = GPT2Config.from_json_file(args.config_file)
    config.architectures = ['GPT2LMHeadModel']
    print('Converting')
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)
    if ds_args is not None:
        tokenizer_type = ds_args.tokenizer_type
        if tokenizer_type == 'GPT2BPETokenizer':
            tokenizer_model_name = 'gpt2'
        elif tokenizer_type == 'PretrainedFromHF':
            tokenizer_model_name = ds_args.tokenizer_name_or_path
        else:
            raise ValueError(f'Unrecognized tokenizer_type {tokenizer_type}')
    else:
        tokenizer_model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class
    print('Saving config')
    config.save_pretrained(basename)
    print(f'Adding {tokenizer_class} tokenizer files')
    tokenizer.save_pretrained(basename)
    output_checkpoint_file = os.path.join(basename, 'pytorch_model.bin')
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)
if __name__ == '__main__':
    main()