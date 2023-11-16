"""Convert Bort checkpoint."""
import argparse
import os
import gluonnlp as nlp
import mxnet as mx
import numpy as np
import torch
from gluonnlp.base import get_home_dir
from gluonnlp.model.bert import BERTEncoder
from gluonnlp.model.utils import _load_vocab
from gluonnlp.vocab import Vocab
from packaging import version
from torch import nn
from transformers import BertConfig, BertForMaskedLM, BertModel, RobertaTokenizer
from transformers.models.bert.modeling_bert import BertIntermediate, BertLayer, BertOutput, BertSelfAttention, BertSelfOutput
from transformers.utils import logging
if version.parse(nlp.__version__) != version.parse('0.8.3'):
    raise Exception('requires gluonnlp == 0.8.3')
if version.parse(mx.__version__) != version.parse('1.5.0'):
    raise Exception('requires mxnet == 1.5.0')
logging.set_verbosity_info()
logger = logging.get_logger(__name__)
SAMPLE_TEXT = 'The Nymphenburg Palace is a beautiful palace in Munich!'

def convert_bort_checkpoint_to_pytorch(bort_checkpoint_path: str, pytorch_dump_folder_path: str):
    if False:
        return 10
    '\n    Convert the original Bort checkpoint (based on MXNET and Gluonnlp) to our BERT structure-\n    '
    bort_4_8_768_1024_hparams = {'attention_cell': 'multi_head', 'num_layers': 4, 'units': 1024, 'hidden_size': 768, 'max_length': 512, 'num_heads': 8, 'scaled': True, 'dropout': 0.1, 'use_residual': True, 'embed_size': 1024, 'embed_dropout': 0.1, 'word_embed': None, 'layer_norm_eps': 1e-05, 'token_type_vocab_size': 2}
    predefined_args = bort_4_8_768_1024_hparams
    encoder = BERTEncoder(attention_cell=predefined_args['attention_cell'], num_layers=predefined_args['num_layers'], units=predefined_args['units'], hidden_size=predefined_args['hidden_size'], max_length=predefined_args['max_length'], num_heads=predefined_args['num_heads'], scaled=predefined_args['scaled'], dropout=predefined_args['dropout'], output_attention=False, output_all_encodings=False, use_residual=predefined_args['use_residual'], activation=predefined_args.get('activation', 'gelu'), layer_norm_eps=predefined_args.get('layer_norm_eps', None))
    vocab_name = 'openwebtext_ccnews_stories_books_cased'
    gluon_cache_dir = os.path.join(get_home_dir(), 'models')
    bort_vocab = _load_vocab(vocab_name, None, gluon_cache_dir, cls=Vocab)
    original_bort = nlp.model.BERTModel(encoder, len(bort_vocab), units=predefined_args['units'], embed_size=predefined_args['embed_size'], embed_dropout=predefined_args['embed_dropout'], word_embed=predefined_args['word_embed'], use_pooler=False, use_token_type_embed=False, token_type_vocab_size=predefined_args['token_type_vocab_size'], use_classifier=False, use_decoder=False)
    original_bort.load_parameters(bort_checkpoint_path, cast_dtype=True, ignore_extra=True)
    params = original_bort._collect_params_with_prefix()
    hf_bort_config_json = {'architectures': ['BertForMaskedLM'], 'attention_probs_dropout_prob': predefined_args['dropout'], 'hidden_act': 'gelu', 'hidden_dropout_prob': predefined_args['dropout'], 'hidden_size': predefined_args['embed_size'], 'initializer_range': 0.02, 'intermediate_size': predefined_args['hidden_size'], 'layer_norm_eps': predefined_args['layer_norm_eps'], 'max_position_embeddings': predefined_args['max_length'], 'model_type': 'bort', 'num_attention_heads': predefined_args['num_heads'], 'num_hidden_layers': predefined_args['num_layers'], 'pad_token_id': 1, 'type_vocab_size': 1, 'vocab_size': len(bort_vocab)}
    hf_bort_config = BertConfig.from_dict(hf_bort_config_json)
    hf_bort_model = BertForMaskedLM(hf_bort_config)
    hf_bort_model.eval()

    def to_torch(mx_array) -> nn.Parameter:
        if False:
            i = 10
            return i + 15
        return nn.Parameter(torch.FloatTensor(mx_array.data().asnumpy()))

    def check_and_map_params(hf_param, gluon_param):
        if False:
            print('Hello World!')
        shape_hf = hf_param.shape
        gluon_param = to_torch(params[gluon_param])
        shape_gluon = gluon_param.shape
        assert shape_hf == shape_gluon, f'The gluon parameter {gluon_param} has shape {shape_gluon}, but expects shape {shape_hf} for Transformers'
        return gluon_param
    hf_bort_model.bert.embeddings.word_embeddings.weight = check_and_map_params(hf_bort_model.bert.embeddings.word_embeddings.weight, 'word_embed.0.weight')
    hf_bort_model.bert.embeddings.position_embeddings.weight = check_and_map_params(hf_bort_model.bert.embeddings.position_embeddings.weight, 'encoder.position_weight')
    hf_bort_model.bert.embeddings.LayerNorm.bias = check_and_map_params(hf_bort_model.bert.embeddings.LayerNorm.bias, 'encoder.layer_norm.beta')
    hf_bort_model.bert.embeddings.LayerNorm.weight = check_and_map_params(hf_bort_model.bert.embeddings.LayerNorm.weight, 'encoder.layer_norm.gamma')
    hf_bort_model.bert.embeddings.token_type_embeddings.weight.data = torch.zeros_like(hf_bort_model.bert.embeddings.token_type_embeddings.weight.data)
    for i in range(hf_bort_config.num_hidden_layers):
        layer: BertLayer = hf_bort_model.bert.encoder.layer[i]
        self_attn: BertSelfAttention = layer.attention.self
        self_attn.key.bias.data = check_and_map_params(self_attn.key.bias.data, f'encoder.transformer_cells.{i}.attention_cell.proj_key.bias')
        self_attn.key.weight.data = check_and_map_params(self_attn.key.weight.data, f'encoder.transformer_cells.{i}.attention_cell.proj_key.weight')
        self_attn.query.bias.data = check_and_map_params(self_attn.query.bias.data, f'encoder.transformer_cells.{i}.attention_cell.proj_query.bias')
        self_attn.query.weight.data = check_and_map_params(self_attn.query.weight.data, f'encoder.transformer_cells.{i}.attention_cell.proj_query.weight')
        self_attn.value.bias.data = check_and_map_params(self_attn.value.bias.data, f'encoder.transformer_cells.{i}.attention_cell.proj_value.bias')
        self_attn.value.weight.data = check_and_map_params(self_attn.value.weight.data, f'encoder.transformer_cells.{i}.attention_cell.proj_value.weight')
        self_output: BertSelfOutput = layer.attention.output
        self_output.dense.bias = check_and_map_params(self_output.dense.bias, f'encoder.transformer_cells.{i}.proj.bias')
        self_output.dense.weight = check_and_map_params(self_output.dense.weight, f'encoder.transformer_cells.{i}.proj.weight')
        self_output.LayerNorm.bias = check_and_map_params(self_output.LayerNorm.bias, f'encoder.transformer_cells.{i}.layer_norm.beta')
        self_output.LayerNorm.weight = check_and_map_params(self_output.LayerNorm.weight, f'encoder.transformer_cells.{i}.layer_norm.gamma')
        intermediate: BertIntermediate = layer.intermediate
        intermediate.dense.bias = check_and_map_params(intermediate.dense.bias, f'encoder.transformer_cells.{i}.ffn.ffn_1.bias')
        intermediate.dense.weight = check_and_map_params(intermediate.dense.weight, f'encoder.transformer_cells.{i}.ffn.ffn_1.weight')
        bert_output: BertOutput = layer.output
        bert_output.dense.bias = check_and_map_params(bert_output.dense.bias, f'encoder.transformer_cells.{i}.ffn.ffn_2.bias')
        bert_output.dense.weight = check_and_map_params(bert_output.dense.weight, f'encoder.transformer_cells.{i}.ffn.ffn_2.weight')
        bert_output.LayerNorm.bias = check_and_map_params(bert_output.LayerNorm.bias, f'encoder.transformer_cells.{i}.ffn.layer_norm.beta')
        bert_output.LayerNorm.weight = check_and_map_params(bert_output.LayerNorm.weight, f'encoder.transformer_cells.{i}.ffn.layer_norm.gamma')
    hf_bort_model.half()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    input_ids = tokenizer.encode_plus(SAMPLE_TEXT)['input_ids']
    gluon_input_ids = mx.nd.array([input_ids])
    output_gluon = original_bort(inputs=gluon_input_ids, token_types=[])
    hf_bort_model.save_pretrained(pytorch_dump_folder_path)
    hf_bort_model = BertModel.from_pretrained(pytorch_dump_folder_path)
    hf_bort_model.eval()
    input_ids = tokenizer.encode_plus(SAMPLE_TEXT, return_tensors='pt')
    output_hf = hf_bort_model(**input_ids)[0]
    gluon_layer = output_gluon[0].asnumpy()
    hf_layer = output_hf[0].detach().numpy()
    max_absolute_diff = np.max(np.abs(hf_layer - gluon_layer)).item()
    success = np.allclose(gluon_layer, hf_layer, atol=0.001)
    if success:
        print('✔️ Both model do output the same tensors')
    else:
        print('❌ Both model do **NOT** output the same tensors')
        print('Absolute difference is:', max_absolute_diff)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bort_checkpoint_path', default=None, type=str, required=True, help='Path the official Bort params file.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_bort_checkpoint_to_pytorch(args.bort_checkpoint_path, args.pytorch_dump_folder_path)