""" File for loading the Pop2Piano model weights from the official repository and to show how tokenizer vocab was
 constructed"""
import json
import torch
from transformers import Pop2PianoConfig, Pop2PianoForConditionalGeneration
official_weights = torch.load('./model-1999-val_0.67311615.ckpt')
state_dict = {}
cfg = Pop2PianoConfig.from_pretrained('sweetcocoa/pop2piano')
model = Pop2PianoForConditionalGeneration(cfg)
state_dict['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'] = official_weights['state_dict']['transformer.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
state_dict['decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'] = official_weights['state_dict']['transformer.decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
state_dict['encoder.embed_tokens.weight'] = official_weights['state_dict']['transformer.encoder.embed_tokens.weight']
state_dict['decoder.embed_tokens.weight'] = official_weights['state_dict']['transformer.decoder.embed_tokens.weight']
state_dict['encoder.final_layer_norm.weight'] = official_weights['state_dict']['transformer.encoder.final_layer_norm.weight']
state_dict['decoder.final_layer_norm.weight'] = official_weights['state_dict']['transformer.decoder.final_layer_norm.weight']
state_dict['lm_head.weight'] = official_weights['state_dict']['transformer.lm_head.weight']
state_dict['mel_conditioner.embedding.weight'] = official_weights['state_dict']['mel_conditioner.embedding.weight']
state_dict['shared.weight'] = official_weights['state_dict']['transformer.shared.weight']
for i in range(cfg.num_layers):
    state_dict[f'encoder.block.{i}.layer.0.SelfAttention.q.weight'] = official_weights['state_dict'][f'transformer.encoder.block.{i}.layer.0.SelfAttention.q.weight']
    state_dict[f'encoder.block.{i}.layer.0.SelfAttention.k.weight'] = official_weights['state_dict'][f'transformer.encoder.block.{i}.layer.0.SelfAttention.k.weight']
    state_dict[f'encoder.block.{i}.layer.0.SelfAttention.v.weight'] = official_weights['state_dict'][f'transformer.encoder.block.{i}.layer.0.SelfAttention.v.weight']
    state_dict[f'encoder.block.{i}.layer.0.SelfAttention.o.weight'] = official_weights['state_dict'][f'transformer.encoder.block.{i}.layer.0.SelfAttention.o.weight']
    state_dict[f'encoder.block.{i}.layer.0.layer_norm.weight'] = official_weights['state_dict'][f'transformer.encoder.block.{i}.layer.0.layer_norm.weight']
    state_dict[f'encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight'] = official_weights['state_dict'][f'transformer.encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight']
    state_dict[f'encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight'] = official_weights['state_dict'][f'transformer.encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight']
    state_dict[f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight'] = official_weights['state_dict'][f'transformer.encoder.block.{i}.layer.1.DenseReluDense.wo.weight']
    state_dict[f'encoder.block.{i}.layer.1.layer_norm.weight'] = official_weights['state_dict'][f'transformer.encoder.block.{i}.layer.1.layer_norm.weight']
for i in range(6):
    state_dict[f'decoder.block.{i}.layer.0.SelfAttention.q.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.0.SelfAttention.q.weight']
    state_dict[f'decoder.block.{i}.layer.0.SelfAttention.k.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.0.SelfAttention.k.weight']
    state_dict[f'decoder.block.{i}.layer.0.SelfAttention.v.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.0.SelfAttention.v.weight']
    state_dict[f'decoder.block.{i}.layer.0.SelfAttention.o.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.0.SelfAttention.o.weight']
    state_dict[f'decoder.block.{i}.layer.0.layer_norm.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.0.layer_norm.weight']
    state_dict[f'decoder.block.{i}.layer.1.EncDecAttention.q.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.1.EncDecAttention.q.weight']
    state_dict[f'decoder.block.{i}.layer.1.EncDecAttention.k.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.1.EncDecAttention.k.weight']
    state_dict[f'decoder.block.{i}.layer.1.EncDecAttention.v.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.1.EncDecAttention.v.weight']
    state_dict[f'decoder.block.{i}.layer.1.EncDecAttention.o.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.1.EncDecAttention.o.weight']
    state_dict[f'decoder.block.{i}.layer.1.layer_norm.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.1.layer_norm.weight']
    state_dict[f'decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight']
    state_dict[f'decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight']
    state_dict[f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.2.DenseReluDense.wo.weight']
    state_dict[f'decoder.block.{i}.layer.2.layer_norm.weight'] = official_weights['state_dict'][f'transformer.decoder.block.{i}.layer.2.layer_norm.weight']
model.load_state_dict(state_dict, strict=True)
torch.save(state_dict, './pytorch_model.bin')

def tokenize(idx, token_type, n_special=4, n_note=128, n_velocity=2):
    if False:
        for i in range(10):
            print('nop')
    if token_type == 'TOKEN_TIME':
        return n_special + n_note + n_velocity + idx
    elif token_type == 'TOKEN_VELOCITY':
        return n_special + n_note + idx
    elif token_type == 'TOKEN_NOTE':
        return n_special + idx
    elif token_type == 'TOKEN_SPECIAL':
        return idx
    else:
        return -1

def detokenize(idx, n_special=4, n_note=128, n_velocity=2, time_idx_offset=0):
    if False:
        return 10
    if idx >= n_special + n_note + n_velocity:
        return ('TOKEN_TIME', idx - (n_special + n_note + n_velocity) + time_idx_offset)
    elif idx >= n_special + n_note:
        return ('TOKEN_VELOCITY', idx - (n_special + n_note))
    elif idx >= n_special:
        return ('TOKEN_NOTE', idx - n_special)
    else:
        return ('TOKEN_SPECIAL', idx)
decoder = {}
for i in range(cfg.vocab_size):
    decoder.update({i: f'{detokenize(i)[1]}_{detokenize(i)[0]}'})
encoder = {v: k for (k, v) in decoder.items()}
with open('./vocab.json', 'w') as file:
    file.write(json.dumps(encoder))