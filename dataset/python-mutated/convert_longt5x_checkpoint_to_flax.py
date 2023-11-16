"""Convert T5/LongT5X checkpoints from the original repository to JAX/FLAX model. This script is an extension of
'src/transformers/models/t5/convert_t5x_checkpoint_to_flax.
"""
import argparse
from t5x import checkpoints
from transformers import AutoConfig, FlaxAutoModelForSeq2SeqLM

def convert_t5x_checkpoint_to_flax(t5x_checkpoint_path, config_name, flax_dump_folder_path):
    if False:
        print('Hello World!')
    config = AutoConfig.from_pretrained(config_name)
    flax_model = FlaxAutoModelForSeq2SeqLM.from_config(config=config)
    t5x_model = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    split_mlp_wi = 'wi_0' in t5x_model['target']['encoder']['layers_0']['mlp']
    if config.model_type == 't5':
        encoder_attn_name = 'SelfAttention'
    if config.model_type == 'longt5' and config.encoder_attention_type == 'local':
        encoder_attn_name = 'LocalSelfAttention'
    elif config.model_type == 'longt5' and config.encoder_attention_type == 'transient-global':
        encoder_attn_name = 'TransientGlobalSelfAttention'
    else:
        raise ValueError("Given config is expected to have `model_type='t5'`, or `model_type='longt5` with `encoder_attention_type` attribute with a value from ['local', 'transient-global].")
    for layer_index in range(config.num_layers):
        layer_name = f'layers_{str(layer_index)}'
        t5x_attention_key = t5x_model['target']['encoder'][layer_name]['attention']['key']['kernel']
        t5x_attention_out = t5x_model['target']['encoder'][layer_name]['attention']['out']['kernel']
        t5x_attention_query = t5x_model['target']['encoder'][layer_name]['attention']['query']['kernel']
        t5x_attention_value = t5x_model['target']['encoder'][layer_name]['attention']['value']['kernel']
        if config.model_type == 'longt5' and config.encoder_attention_type == 'transient-global':
            t5x_global_layer_norm = t5x_model['target']['encoder'][layer_name]['attention']['T5LayerNorm_0']['scale']
        t5x_attention_layer_norm = t5x_model['target']['encoder'][layer_name]['pre_attention_layer_norm']['scale']
        if split_mlp_wi:
            t5x_mlp_wi_0 = t5x_model['target']['encoder'][layer_name]['mlp']['wi_0']['kernel']
            t5x_mlp_wi_1 = t5x_model['target']['encoder'][layer_name]['mlp']['wi_1']['kernel']
        else:
            t5x_mlp_wi = t5x_model['target']['encoder'][layer_name]['mlp']['wi']['kernel']
        t5x_mlp_wo = t5x_model['target']['encoder'][layer_name]['mlp']['wo']['kernel']
        t5x_mlp_layer_norm = t5x_model['target']['encoder'][layer_name]['pre_mlp_layer_norm']['scale']
        flax_model_encoder_layer_block = flax_model.params['encoder']['block'][str(layer_index)]['layer']
        flax_model_encoder_layer_block['0'][encoder_attn_name]['k']['kernel'] = t5x_attention_key
        flax_model_encoder_layer_block['0'][encoder_attn_name]['o']['kernel'] = t5x_attention_out
        flax_model_encoder_layer_block['0'][encoder_attn_name]['q']['kernel'] = t5x_attention_query
        flax_model_encoder_layer_block['0'][encoder_attn_name]['v']['kernel'] = t5x_attention_value
        flax_model_encoder_layer_block['0']['layer_norm']['weight'] = t5x_attention_layer_norm
        if config.model_type == 'longt5' and config.encoder_attention_type == 'transient-global':
            flax_model_encoder_layer_block['0'][encoder_attn_name]['global_input_layer_norm']['weight'] = t5x_global_layer_norm
        if split_mlp_wi:
            flax_model_encoder_layer_block['1']['DenseReluDense']['wi_0']['kernel'] = t5x_mlp_wi_0
            flax_model_encoder_layer_block['1']['DenseReluDense']['wi_1']['kernel'] = t5x_mlp_wi_1
        else:
            flax_model_encoder_layer_block['1']['DenseReluDense']['wi']['kernel'] = t5x_mlp_wi
        flax_model_encoder_layer_block['1']['DenseReluDense']['wo']['kernel'] = t5x_mlp_wo
        flax_model_encoder_layer_block['1']['layer_norm']['weight'] = t5x_mlp_layer_norm
        flax_model.params['encoder']['block'][str(layer_index)]['layer'] = flax_model_encoder_layer_block
    t5x_encoder_rel_embedding = t5x_model['target']['encoder']['relpos_bias']['rel_embedding'].T
    flax_model.params['encoder']['block']['0']['layer']['0'][encoder_attn_name]['relative_attention_bias']['embedding'] = t5x_encoder_rel_embedding
    if config.model_type == 'longt5' and config.encoder_attention_type == 'transient-global':
        t5x_encoder_global_rel_embedding = t5x_model['target']['encoder']['side_relpos_bias']['rel_embedding'].T
        flax_model.params['encoder']['block']['0']['layer']['0'][encoder_attn_name]['global_relative_attention_bias']['embedding'] = t5x_encoder_global_rel_embedding
    t5x_encoder_norm = t5x_model['target']['encoder']['encoder_norm']['scale']
    flax_model.params['encoder']['final_layer_norm']['weight'] = t5x_encoder_norm
    for layer_index in range(config.num_layers):
        layer_name = f'layers_{str(layer_index)}'
        t5x_attention_key = t5x_model['target']['decoder'][layer_name]['self_attention']['key']['kernel']
        t5x_attention_out = t5x_model['target']['decoder'][layer_name]['self_attention']['out']['kernel']
        t5x_attention_query = t5x_model['target']['decoder'][layer_name]['self_attention']['query']['kernel']
        t5x_attention_value = t5x_model['target']['decoder'][layer_name]['self_attention']['value']['kernel']
        t5x_pre_attention_layer_norm = t5x_model['target']['decoder'][layer_name]['pre_self_attention_layer_norm']['scale']
        t5x_enc_dec_attention_module = t5x_model['target']['decoder'][layer_name]['encoder_decoder_attention']
        t5x_enc_dec_attention_key = t5x_enc_dec_attention_module['key']['kernel']
        t5x_enc_dec_attention_out = t5x_enc_dec_attention_module['out']['kernel']
        t5x_enc_dec_attention_query = t5x_enc_dec_attention_module['query']['kernel']
        t5x_enc_dec_attention_value = t5x_enc_dec_attention_module['value']['kernel']
        t5x_cross_layer_norm = t5x_model['target']['decoder'][layer_name]['pre_cross_attention_layer_norm']['scale']
        if split_mlp_wi:
            t5x_mlp_wi_0 = t5x_model['target']['decoder'][layer_name]['mlp']['wi_0']['kernel']
            t5x_mlp_wi_1 = t5x_model['target']['decoder'][layer_name]['mlp']['wi_1']['kernel']
        else:
            t5x_mlp_wi = t5x_model['target']['decoder'][layer_name]['mlp']['wi']['kernel']
        t5x_mlp_wo = t5x_model['target']['decoder'][layer_name]['mlp']['wo']['kernel']
        tx5_mlp_layer_norm = t5x_model['target']['decoder'][layer_name]['pre_mlp_layer_norm']['scale']
        flax_model_decoder_layer_block = flax_model.params['decoder']['block'][str(layer_index)]['layer']
        flax_model_decoder_layer_block['0']['SelfAttention']['k']['kernel'] = t5x_attention_key
        flax_model_decoder_layer_block['0']['SelfAttention']['o']['kernel'] = t5x_attention_out
        flax_model_decoder_layer_block['0']['SelfAttention']['q']['kernel'] = t5x_attention_query
        flax_model_decoder_layer_block['0']['SelfAttention']['v']['kernel'] = t5x_attention_value
        flax_model_decoder_layer_block['0']['layer_norm']['weight'] = t5x_pre_attention_layer_norm
        flax_model_decoder_layer_block['1']['EncDecAttention']['k']['kernel'] = t5x_enc_dec_attention_key
        flax_model_decoder_layer_block['1']['EncDecAttention']['o']['kernel'] = t5x_enc_dec_attention_out
        flax_model_decoder_layer_block['1']['EncDecAttention']['q']['kernel'] = t5x_enc_dec_attention_query
        flax_model_decoder_layer_block['1']['EncDecAttention']['v']['kernel'] = t5x_enc_dec_attention_value
        flax_model_decoder_layer_block['1']['layer_norm']['weight'] = t5x_cross_layer_norm
        if split_mlp_wi:
            flax_model_decoder_layer_block['2']['DenseReluDense']['wi_0']['kernel'] = t5x_mlp_wi_0
            flax_model_decoder_layer_block['2']['DenseReluDense']['wi_1']['kernel'] = t5x_mlp_wi_1
        else:
            flax_model_decoder_layer_block['2']['DenseReluDense']['wi']['kernel'] = t5x_mlp_wi
        flax_model_decoder_layer_block['2']['DenseReluDense']['wo']['kernel'] = t5x_mlp_wo
        flax_model_decoder_layer_block['2']['layer_norm']['weight'] = tx5_mlp_layer_norm
        flax_model.params['decoder']['block'][str(layer_index)]['layer'] = flax_model_decoder_layer_block
    tx5_decoder_norm = t5x_model['target']['decoder']['decoder_norm']['scale']
    flax_model.params['decoder']['final_layer_norm']['weight'] = tx5_decoder_norm
    t5x_decoder_rel_embedding = t5x_model['target']['decoder']['relpos_bias']['rel_embedding'].T
    flax_model.params['decoder']['block']['0']['layer']['0']['SelfAttention']['relative_attention_bias']['embedding'] = t5x_decoder_rel_embedding
    tx5_token_embeddings = t5x_model['target']['token_embedder']['embedding']
    flax_model.params['shared']['embedding'] = tx5_token_embeddings
    if 'logits_dense' in t5x_model['target']['decoder']:
        flax_model.params['lm_head']['kernel'] = t5x_model['target']['decoder']['logits_dense']['kernel']
    flax_model.save_pretrained(flax_dump_folder_path)
    print('T5X Model was sucessfully converted!')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t5x_checkpoint_path', default=None, type=str, required=True, help='Path the T5X checkpoint.')
    parser.add_argument('--config_name', default=None, type=str, required=True, help='Config name of LongT5/T5 model.')
    parser.add_argument('--flax_dump_folder_path', default=None, type=str, required=True, help='Path to the output FLAX model.')
    args = parser.parse_args()
    convert_t5x_checkpoint_to_flax(args.t5x_checkpoint_path, args.config_name, args.flax_dump_folder_path)