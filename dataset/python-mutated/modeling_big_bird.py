""" PyTorch BigBird model."""
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, CausalLMOutputWithCrossAttentions, MaskedLMOutput, MultipleChoiceModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_big_bird import BigBirdConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'google/bigbird-roberta-base'
_CONFIG_FOR_DOC = 'BigBirdConfig'
BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST = ['google/bigbird-roberta-base', 'google/bigbird-roberta-large', 'google/bigbird-base-trivia-itc']
_TRIVIA_QA_MAPPING = {'big_bird_attention': 'attention/self', 'output_layer_norm': 'output/LayerNorm', 'attention_output': 'attention/output/dense', 'output': 'output/dense', 'self_attention_layer_norm': 'attention/output/LayerNorm', 'intermediate': 'intermediate/dense', 'word_embeddings': 'bert/embeddings/word_embeddings', 'position_embedding': 'bert/embeddings/position_embeddings', 'type_embeddings': 'bert/embeddings/token_type_embeddings', 'embeddings': 'bert/embeddings', 'layer_normalization': 'output/LayerNorm', 'layer_norm': 'LayerNorm', 'trivia_qa_head': 'qa_classifier', 'dense': 'intermediate/dense', 'dense_1': 'qa_outputs'}

def load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=False):
    if False:
        while True:
            i = 10
    'Load tf checkpoints in a pytorch model.'

    def load_tf_weights_bert(init_vars, tf_path):
        if False:
            i = 10
            return i + 15
        names = []
        tf_weights = {}
        for (name, shape) in init_vars:
            array = tf.train.load_variable(tf_path, name)
            name = name.replace('bert/encoder/LayerNorm', 'bert/embeddings/LayerNorm')
            logger.info(f'Loading TF weight {name} with shape {shape}')
            names.append(name)
            tf_weights[name] = array
        return (names, tf_weights)

    def load_tf_weights_trivia_qa(init_vars):
        if False:
            i = 10
            return i + 15
        names = []
        tf_weights = {}
        for (i, var) in enumerate(init_vars):
            name_items = var.name.split('/')
            if 'transformer_scaffold' in name_items[0]:
                layer_name_items = name_items[0].split('_')
                if len(layer_name_items) < 3:
                    layer_name_items += [0]
                name_items[0] = f'bert/encoder/layer_{layer_name_items[2]}'
            name = '/'.join([_TRIVIA_QA_MAPPING[x] if x in _TRIVIA_QA_MAPPING else x for x in name_items])[:-2]
            if 'self/attention/output' in name:
                name = name.replace('self/attention/output', 'output')
            if i >= len(init_vars) - 2:
                name = name.replace('intermediate', 'output')
            logger.info(f'Loading TF weight {name} with shape {var.shape}')
            array = var.value().numpy()
            names.append(name)
            tf_weights[name] = array
        return (names, tf_weights)
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f'Converting TensorFlow checkpoint from {tf_path}')
    init_vars = tf.saved_model.load(tf_path).variables if is_trivia_qa else tf.train.list_variables(tf_path)
    if len(init_vars) <= 0:
        raise ValueError('Loaded trained variables cannot be empty.')
    pt_names = list(model.state_dict().keys())
    if is_trivia_qa:
        (names, tf_weights) = load_tf_weights_trivia_qa(init_vars)
    else:
        (names, tf_weights) = load_tf_weights_bert(init_vars, tf_path)
    for txt_name in names:
        array = tf_weights[txt_name]
        name = txt_name.split('/')
        if any((n in ['adam_v', 'adam_m', 'AdamWeightDecayOptimizer', 'AdamWeightDecayOptimizer_1', 'global_step'] for n in name)):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        pt_name = []
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                scope_names = re.split('_(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
                pt_name.append('weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
                pt_name.append('bias')
            elif scope_names[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
                pt_name.append('weight')
            elif scope_names[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
                pt_name.append('classifier')
            elif scope_names[0] == 'transform':
                pointer = getattr(pointer, 'transform')
                pt_name.append('transform')
                if 'bias' in name or 'kernel' in name:
                    pointer = getattr(pointer, 'dense')
                    pt_name.append('dense')
                elif 'beta' in name or 'gamma' in name:
                    pointer = getattr(pointer, 'LayerNorm')
                    pt_name.append('LayerNorm')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                    pt_name.append(f'{scope_names[0]}')
                except AttributeError:
                    logger.info(f'Skipping {m_name}')
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
                pt_name.append(f'{num}')
        if m_name[-11:] == '_embeddings' or m_name == 'embeddings':
            pointer = getattr(pointer, 'weight')
            pt_name.append('weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            if len(array.shape) > len(pointer.shape) and math.prod(array.shape) == math.prod(pointer.shape):
                if txt_name.endswith('attention/self/key/kernel') or txt_name.endswith('attention/self/query/kernel') or txt_name.endswith('attention/self/value/kernel'):
                    array = array.transpose(1, 0, 2).reshape(pointer.shape)
                elif txt_name.endswith('attention/output/dense/kernel'):
                    array = array.transpose(0, 2, 1).reshape(pointer.shape)
                else:
                    array = array.reshape(pointer.shape)
            if pointer.shape != array.shape:
                raise ValueError(f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched of {txt_name}.')
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pt_weight_name = '.'.join(pt_name)
        logger.info(f'Initialize PyTorch weight {pt_weight_name} from {txt_name}.')
        pointer.data = torch.from_numpy(array)
        tf_weights.pop(txt_name, None)
        pt_names.remove(pt_weight_name)
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    logger.info(f"Weights not initialized in PyTorch model: {', '.join(pt_names)}.")
    return model

class BigBirdEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.register_buffer('token_type_ids', torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)
        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if False:
            for i in range(10):
                print('nop')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * self.hidden_size ** 0.5
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings

class BigBirdSelfAttention(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        if False:
            return 10
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        if False:
            return 10
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class BigBirdBlockSparseAttention(nn.Module):

    def __init__(self, config, seed=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.max_seqlen = config.max_position_embeddings
        self.seed = seed
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}.')
        self.num_attention_heads = config.num_attention_heads
        self.num_random_blocks = config.num_random_blocks
        self.block_size = config.block_size
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

    def transpose_for_scores(self, x):
        if False:
            return 10
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, band_mask=None, from_mask=None, to_mask=None, from_blocked_mask=None, to_blocked_mask=None, output_attentions=None):
        if False:
            print('Hello World!')
        (batch_size, seqlen, _) = hidden_states.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size
        if from_seq_length % from_block_size != 0:
            raise ValueError('Query sided sequence length must be multiple of block size')
        if to_seq_length % to_block_size != 0:
            raise ValueError('Key/Value sided sequence length must be multiple of block size')
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        (context_layer, attention_probs) = self.bigbird_block_sparse_attention(query_layer, key_layer, value_layer, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, self.num_attention_heads, self.num_random_blocks, self.attention_head_size, from_block_size, to_block_size, batch_size, from_seq_length, to_seq_length, seed=self.seed, plan_from_length=None, plan_num_rand_blocks=None, output_attentions=output_attentions)
        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=None):
        if False:
            i = 10
            return i + 15
        'Fast nd matrix multiplication'
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(inp_1.shape[:ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1]))

    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
        if False:
            for i in range(10):
                print('nop')
        'Fast nd matrix multiplication with transpose'
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2)).view(inp_1.shape[:ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))

    def bigbird_block_sparse_attention(self, query_layer, key_layer, value_layer, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, n_heads, n_rand_blocks, attention_head_size, from_block_size, to_block_size, batch_size, from_seq_len, to_seq_len, seed, plan_from_length, plan_num_rand_blocks, output_attentions):
        if False:
            i = 10
            return i + 15
        if from_seq_len // from_block_size != to_seq_len // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        rsqrt_d = 1 / math.sqrt(attention_head_size)
        bsz = batch_size
        attn_mask_penalty = -10000.0
        np.random.seed(seed)
        if from_seq_len in [1024, 3072, 4096]:
            rand_attn = [self._bigbird_block_rand_mask(self.max_seqlen, self.max_seqlen, from_block_size, to_block_size, n_rand_blocks, last_idx=1024)[:from_seq_len // from_block_size - 2] for _ in range(n_heads)]
        else:
            if plan_from_length is None:
                (plan_from_length, plan_num_rand_blocks) = self._get_rand_attn_plan(from_seq_len, from_block_size, n_rand_blocks)
            rand_attn = self._bigbird_block_rand_mask_with_head(from_seq_length=from_seq_len, to_seq_length=to_seq_len, from_block_size=from_block_size, to_block_size=to_block_size, num_heads=n_heads, plan_from_length=plan_from_length, plan_num_rand_blocks=plan_num_rand_blocks)
        rand_attn = np.stack(rand_attn, axis=0)
        rand_attn = torch.tensor(rand_attn, device=query_layer.device, dtype=torch.long)
        rand_attn.unsqueeze_(0)
        rand_attn = torch.cat([rand_attn for _ in range(batch_size)], dim=0)
        rand_mask = self._create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn, n_heads, n_rand_blocks, bsz, from_seq_len, from_block_size)
        blocked_query_matrix = query_layer.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1)
        blocked_key_matrix = key_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        blocked_value_matrix = value_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)
        gathered_key = gathered_key.view(bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1)
        gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
        gathered_value = gathered_value.view(bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1)
        first_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 0], key_layer, ndim=4)
        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = nn.functional.softmax(first_product, dim=-1)
        first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
        first_context_layer.unsqueeze_(2)
        second_key_mat = torch.cat([blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1], blocked_key_matrix[:, :, 2], blocked_key_matrix[:, :, -1], gathered_key[:, :, 0]], dim=2)
        second_value_mat = torch.cat([blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1], blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1], gathered_value[:, :, 0]], dim=2)
        second_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 1], second_key_mat, ndim=4)
        second_seq_pad = torch.cat([to_mask[:, :, :, :3 * to_block_size], to_mask[:, :, :, -to_block_size:], to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size])], dim=3)
        second_rand_pad = torch.cat([rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]), rand_mask[:, :, 0]], dim=3)
        second_product = second_product * rsqrt_d
        second_product += (1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = nn.functional.softmax(second_product, dim=-1)
        second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)
        second_context_layer.unsqueeze_(2)
        exp_blocked_key_matrix = torch.cat([blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], dim=3)
        exp_blocked_value_matrix = torch.cat([blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]], dim=3)
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
        inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)
        inner_band_product = inner_band_product * rsqrt_d
        rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)
        rand_band_product = rand_band_product * rsqrt_d
        first_band_product = torch.einsum('bhlqd,bhkd->bhlqk', middle_query_matrix, blocked_key_matrix[:, :, 0])
        first_band_product = first_band_product * rsqrt_d
        last_band_product = torch.einsum('bhlqd,bhkd->bhlqk', middle_query_matrix, blocked_key_matrix[:, :, -1])
        last_band_product = last_band_product * rsqrt_d
        inner_band_product += (1.0 - band_mask) * attn_mask_penalty
        first_band_product += (1.0 - to_mask[:, :, :, :to_block_size].unsqueeze(3)) * attn_mask_penalty
        last_band_product += (1.0 - to_mask[:, :, :, -to_block_size:].unsqueeze(3)) * attn_mask_penalty
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty
        band_product = torch.cat([first_band_product, inner_band_product, rand_band_product, last_band_product], dim=-1)
        attn_weights = nn.functional.softmax(band_product, dim=-1)
        context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :, to_block_size:4 * to_block_size], exp_blocked_value_matrix, ndim=5)
        context_layer += self.torch_bmm_nd(attn_weights[:, :, :, :, 4 * to_block_size:-to_block_size], gathered_value[:, :, 1:-1], ndim=5)
        context_layer += torch.einsum('bhlqk,bhkd->bhlqd', attn_weights[:, :, :, :, :to_block_size], blocked_value_matrix[:, :, 0])
        context_layer += torch.einsum('bhlqk,bhkd->bhlqd', attn_weights[:, :, :, :, -to_block_size:], blocked_value_matrix[:, :, -1])
        second_last_key_mat = torch.cat([blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3], blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1], gathered_key[:, :, -1]], dim=2)
        second_last_value_mat = torch.cat([blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3], blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1], gathered_value[:, :, -1]], dim=2)
        second_last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4)
        second_last_seq_pad = torch.cat([to_mask[:, :, :, :to_block_size], to_mask[:, :, :, -3 * to_block_size:], to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size])], dim=3)
        second_last_rand_pad = torch.cat([rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]), rand_mask[:, :, -1]], dim=3)
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = nn.functional.softmax(second_last_product, dim=-1)
        second_last_context_layer = self.torch_bmm_nd(second_last_attn_weights, second_last_value_mat, ndim=4)
        second_last_context_layer.unsqueeze_(2)
        last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], key_layer, ndim=4)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = nn.functional.softmax(last_product, dim=-1)
        last_context_layer = self.torch_bmm_nd(last_attn_weights, value_layer, ndim=4)
        last_context_layer.unsqueeze_(2)
        context_layer = torch.cat([first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer], dim=2)
        context_layer = context_layer.view((bsz, n_heads, from_seq_len, -1)) * from_mask
        context_layer = torch.transpose(context_layer, 1, 2)
        if output_attentions:
            attention_probs = torch.zeros(bsz, n_heads, from_seq_len, to_seq_len, dtype=torch.float, device=context_layer.device)
            attention_probs[:, :, :from_block_size, :] = first_attn_weights
            attention_probs[:, :, from_block_size:2 * from_block_size, :3 * to_block_size] = second_attn_weights[:, :, :, :3 * to_block_size]
            attention_probs[:, :, from_block_size:2 * from_block_size, -to_block_size:] = second_attn_weights[:, :, :, 3 * to_block_size:4 * to_block_size]
            for (p1, i1, w1) in zip(range(bsz), rand_attn, second_attn_weights):
                for (p2, i2, w2) in zip(range(n_heads), i1, w1):
                    attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)
                    right_slice = w2[:, 4 * to_block_size:]
                    attn_probs_view[p1, p2, 1, :, i2[0]] = right_slice.view(from_block_size, n_rand_blocks, to_block_size)
            for q_idx in range(from_seq_len // from_block_size - 4):
                attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)[:, :, 2:-2, :, 1:-1, :]
                right_slice = attn_weights[:, :, q_idx, :, to_block_size:4 * to_block_size]
                attn_probs_view[:, :, q_idx, :, q_idx:q_idx + 3, :] = right_slice.view(bsz, n_heads, from_block_size, 3, to_block_size)
            attention_probs[:, :, 2 * from_block_size:-2 * from_block_size, :to_block_size] = attn_weights[:, :, :, :, :to_block_size].view(bsz, n_heads, -1, to_block_size)
            attention_probs[:, :, 2 * from_block_size:-2 * from_block_size, -to_block_size:] = attn_weights[:, :, :, :, -to_block_size:].view(bsz, n_heads, -1, to_block_size)
            for (p1, i1, w1) in zip(range(bsz), rand_attn, attn_weights):
                for (p2, i2, w2) in zip(range(n_heads), i1, w1):
                    for q_idx in range(1, len(i2) - 1):
                        attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)
                        right_slice = w2[q_idx - 1, :, 4 * to_block_size:-to_block_size]
                        attn_probs_view[p1, p2, q_idx + 1, :, i2[q_idx]] = right_slice.view(from_block_size, n_rand_blocks, to_block_size)
            attention_probs[:, :, -2 * from_block_size:-from_block_size, :to_block_size] = second_last_attn_weights[:, :, :, :to_block_size]
            attention_probs[:, :, -2 * from_block_size:-from_block_size, -3 * to_block_size:] = second_last_attn_weights[:, :, :, to_block_size:4 * to_block_size]
            for (p1, i1, w1) in zip(range(bsz), rand_attn, second_last_attn_weights):
                for (p2, i2, w2) in zip(range(n_heads), i1, w1):
                    attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)
                    right_slice = w2[:, 4 * to_block_size:]
                    attn_probs_view[p1, p2, -2, :, i2[-1]] = right_slice.view(from_block_size, n_rand_blocks, to_block_size)
            attention_probs[:, :, -from_block_size:, :] = last_attn_weights
        else:
            attention_probs = None
        return (context_layer, attention_probs)

    @staticmethod
    def torch_gather_b2(params, indices):
        if False:
            return 10
        if params.shape[:2] != indices.shape[:2]:
            raise ValueError(f'Make sure that the first two dimensions of params and indices are identical,                 but they are params: {params.shape[:2]} vs. indices: {indices.shape[:2]}')
        num_indices_to_gather = indices.shape[-2] * indices.shape[-1]
        num_indices_to_pick_from = params.shape[2]
        shift = torch.arange(indices.shape[0] * indices.shape[1] * num_indices_to_gather, device=indices.device)
        indices_shift = torch.div(shift, num_indices_to_gather, rounding_mode='floor') * num_indices_to_pick_from
        flattened_indices = indices.view(-1) + indices_shift
        flattened_params = params.reshape(-1, params.shape[-2], params.shape[-1])
        out_flattened = flattened_params.index_select(0, flattened_indices)
        out = out_flattened.reshape(params.shape[:2] + (num_indices_to_gather,) + params.shape[3:])
        return out

    @staticmethod
    def _create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn, num_attention_heads, num_rand_blocks, batch_size, from_seq_length, from_block_size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create 3D attention mask from a 2D tensor mask.\n\n        Args:\n            from_blocked_mask: 2D Tensor of shape [batch_size,\n            from_seq_length//from_block_size, from_block_size].\n            to_blocked_mask: int32 Tensor of shape [batch_size,\n            to_seq_length//to_block_size, to_block_size].\n            rand_attn: [batch_size, num_attention_heads,\n            from_seq_length//from_block_size-2, num_rand_blocks]\n            num_attention_heads: int. Number of attention heads.\n            num_rand_blocks: int. Number of random chunks per row.\n            batch_size: int. Batch size for computation.\n            from_seq_length: int. length of from sequence.\n            from_block_size: int. size of block in from sequence.\n\n        Returns:\n            float Tensor of shape [batch_size, num_attention_heads, from_seq_length//from_block_size-2,\n            from_block_size, num_rand_blocks*to_block_size].\n        '
        num_windows = from_seq_length // from_block_size - 2
        rand_mask = torch.stack([p1[i1.flatten()] for (p1, i1) in zip(to_blocked_mask, rand_attn)])
        rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size)
        rand_mask = torch.einsum('blq,bhlk->bhlqk', from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        if False:
            print('Hello World!')
        '\n        Gives the plan of where to put random attention.\n\n        Args:\n            from_seq_length: int. length of from sequence.\n            from_block_size: int. size of block in from sequence.\n            num_rand_blocks: int. Number of random chunks per row.\n\n        Returns:\n            plan_from_length: ending location of from block plan_num_rand_blocks: number of random ending location for\n            each block\n        '
        plan_from_length = []
        plan_num_rand_blocks = []
        if 2 * num_rand_blocks + 5 < from_seq_length // from_block_size:
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        elif num_rand_blocks + 5 < from_seq_length // from_block_size:
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - num_rand_blocks // 2)
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)
        return (plan_from_length, plan_num_rand_blocks)

    def _bigbird_block_rand_mask(self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1):
        if False:
            return 10
        '\n        Create adjacency list of random attention.\n\n        Args:\n            from_seq_length: int. length of from sequence.\n            to_seq_length: int. length of to sequence.\n            from_block_size: int. size of block in from sequence.\n            to_block_size: int. size of block in to sequence.\n            num_rand_blocks: int. Number of random chunks per row.\n            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,\n            if positive then num_rand_blocks blocks chosen only up to last_idx.\n\n        Returns:\n            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks\n        '
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        if not self.training:
            return rand_attn
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > 2 * to_block_size:
            last = last_idx // to_block_size - 1
        r = num_rand_blocks
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            elif start > last:
                start = last
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            elif end + 1 == last:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            else:
                rand_attn[i - 1, :] = np.random.permutation(np.concatenate((middle_seq[:start], middle_seq[end + 1:last])))[:r]
        return rand_attn

    def _bigbird_block_rand_mask_with_head(self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_heads, plan_from_length, plan_num_rand_blocks, window_block_left=1, window_block_right=1, global_block_top=1, global_block_bottom=1, global_block_left=1, global_block_right=1):
        if False:
            return 10
        '\n        Create adjacency list of random attention.\n\n        Args:\n            from_seq_length: int. length of from sequence.\n            to_seq_length: int. length of to sequence.\n            from_block_size: int. size of block in from sequence.\n            to_block_size: int. size of block in to sequence.\n            num_heads: int. total number of heads.\n            plan_from_length: list. plan from length where num_random_blocks are chosen from.\n            plan_num_rand_blocks: list. number of rand blocks within the plan.\n            window_block_left: int. number of blocks of window to left of a block.\n            window_block_right: int. number of blocks of window to right of a block.\n            global_block_top: int. number of blocks at the top.\n            global_block_bottom: int. number of blocks at the bottom.\n            global_block_left: int. Number of blocks globally used to the left.\n            global_block_right: int. Number of blocks globally used to the right.\n\n        Returns:\n            adjacency list of size num_head where each element is of size from_seq_length//from_block_size-2 by\n            num_rand_blocks\n        '
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        if from_seq_length not in plan_from_length:
            raise ValueError('Error from sequence length not in plan!')
        num_blocks = from_seq_length // from_block_size
        plan_block_length = np.array(plan_from_length) // from_block_size
        max_plan_idx = plan_from_length.index(from_seq_length)
        rand_attn = [np.zeros((num_blocks, np.sum(plan_num_rand_blocks[:max_plan_idx + 1])), dtype=np.int32) for i in range(num_heads)]
        if not self.training:
            for nh in range(num_heads):
                rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks - global_block_bottom, :]
            return rand_attn
        for plan_idx in range(max_plan_idx + 1):
            rnd_r_cnt = 0
            if plan_idx > 0:
                if plan_num_rand_blocks[plan_idx] > 0:
                    rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
                    for blk_rw_idx in range(global_block_top, plan_block_length[plan_idx - 1]):
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=plan_block_length[plan_idx - 1], to_end_block_id=plan_block_length[plan_idx], num_rand_blocks=plan_num_rand_blocks[plan_idx], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right)
                for pl_id in range(plan_idx):
                    if plan_num_rand_blocks[pl_id] == 0:
                        continue
                    for blk_rw_idx in range(plan_block_length[plan_idx - 1], plan_block_length[plan_idx]):
                        rnd_r_cnt = 0
                        to_start_block_id = 0
                        if pl_id > 0:
                            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                            to_start_block_id = plan_block_length[pl_id - 1]
                        curr_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id + 1]))
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=to_start_block_id, to_end_block_id=plan_block_length[pl_id], num_rand_blocks=plan_num_rand_blocks[pl_id], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right)
            if plan_num_rand_blocks[plan_idx] == 0:
                continue
            curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
            from_start_block_id = global_block_top
            to_start_block_id = 0
            if plan_idx > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                from_start_block_id = plan_block_length[plan_idx - 1]
                to_start_block_id = plan_block_length[plan_idx - 1]
            for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
                for h in range(num_heads):
                    rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=to_start_block_id, to_end_block_id=plan_block_length[plan_idx], num_rand_blocks=plan_num_rand_blocks[plan_idx], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right)
        for nh in range(num_heads):
            rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks - global_block_bottom, :]
        return rand_attn

    @staticmethod
    def _get_single_block_row_attention(block_id, to_start_block_id, to_end_block_id, num_rand_blocks, window_block_left=1, window_block_right=1, global_block_left=1, global_block_right=1):
        if False:
            while True:
                i = 10
        '\n        For a single row block get random row attention.\n\n        Args:\n            block_id: int. block id of row.\n            to_start_block_id: int. random attention column start id.\n            to_end_block_id: int. random attention column end id.\n            num_rand_blocks: int. number of random blocks to be selected.\n            window_block_left: int. number of blocks of window to left of a block.\n            window_block_right: int. number of blocks of window to right of a block.\n            global_block_left: int. Number of blocks globally used to the left.\n            global_block_right: int. Number of blocks globally used to the right.\n\n        Returns:\n            row containing the random attention vector of size num_rand_blocks.\n        '
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        perm_block = np.random.permutation(to_block_list)
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)
        selected_random_blokcs = []
        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blokcs.append(perm_block[i])
            if len(selected_random_blokcs) == num_rand_blocks:
                break
        return np.array(selected_random_blokcs, dtype=np.int32)

class BigBirdSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BigBirdAttention(nn.Module):

    def __init__(self, config, seed=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.attention_type = config.attention_type
        self.config = config
        self.seed = seed
        if self.config.attention_type == 'original_full':
            self.self = BigBirdSelfAttention(config)
        elif self.config.attention_type == 'block_sparse':
            self.self = BigBirdBlockSparseAttention(config, seed)
        else:
            raise ValueError(f'attention_type can either be original_full or block_sparse, but is {self.config.attention_type}')
        self.output = BigBirdSelfOutput(config)

    def set_attention_type(self, value: str):
        if False:
            for i in range(10):
                print('nop')
        if value not in ['original_full', 'block_sparse']:
            raise ValueError(f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}")
        if value == self.attention_type:
            return
        self.attention_type = value
        if value == 'original_full':
            attn_weights = BigBirdSelfAttention(self.config)
        else:
            attn_weights = BigBirdBlockSparseAttention(self.config, self.seed)
        attn_weights.query = self.self.query
        attn_weights.value = self.self.value
        attn_weights.key = self.self.key
        self.self = attn_weights
        self.attention_type = value
        if not self.training:
            self.self.eval()

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, band_mask=None, from_mask=None, to_mask=None, from_blocked_mask=None, to_blocked_mask=None):
        if False:
            return 10
        if band_mask is not None:
            band_mask = band_mask.to(hidden_states.dtype)
        if from_mask is not None:
            from_mask = from_mask.to(hidden_states.dtype)
        if to_mask is not None:
            to_mask = to_mask.to(hidden_states.dtype)
        if self.attention_type == 'original_full':
            self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        else:
            if encoder_hidden_states is not None:
                raise ValueError("BigBird cannot be used as a decoder when config.attention_type != 'original_full'")
            self_outputs = self.self(hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class BigBirdIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BigBirdOutput(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BigBirdLayer(nn.Module):

    def __init__(self, config, seed=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BigBirdAttention(config, seed=seed)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = BigBirdAttention(config)
        self.intermediate = BigBirdIntermediate(config)
        self.output = BigBirdOutput(config)

    def set_attention_type(self, value: str):
        if False:
            while True:
                i = 10
        if value not in ['original_full', 'block_sparse']:
            raise ValueError(f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}")
        if value == self.attention_type:
            return
        self.attention_type = value
        self.attention.set_attention_type(value)
        if self.add_cross_attention:
            self.crossattention.set_attention_type(value)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, band_mask=None, from_mask=None, to_mask=None, blocked_encoder_mask=None, past_key_value=None, output_attentions=False):
        if False:
            i = 10
            return i + 15
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=self_attn_past_key_value, output_attentions=output_attentions, band_mask=band_mask, from_mask=from_mask, to_mask=to_mask, from_blocked_mask=blocked_encoder_mask, to_blocked_mask=blocked_encoder_mask)
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise ValueError(f'If `encoder_hidden_states` are passed, {self} has to be instantiated with                     cross-attention layers by setting `config.add_cross_attention=True`')
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, cross_attn_past_key_value, output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        if False:
            while True:
                i = 10
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BigBirdEncoder(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type
        self.layer = nn.ModuleList([BigBirdLayer(config, seed=layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def set_attention_type(self, value: str):
        if False:
            while True:
                i = 10
        if value not in ['original_full', 'block_sparse']:
            raise ValueError(f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}")
        if value == self.attention_type:
            return
        self.attention_type = value
        for layer in self.layer:
            layer.set_attention_type(value)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, band_mask=None, from_mask=None, to_mask=None, blocked_encoder_mask=None, return_dict=True) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]:
        if False:
            return 10
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        next_decoder_cache = () if use_cache else None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, band_mask, from_mask, to_mask, blocked_encoder_mask, past_key_value, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, band_mask, from_mask, to_mask, blocked_encoder_mask, past_key_value, output_attentions)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

class BigBirdPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BigBirdLMPredictionHead(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.transform = BigBirdPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        if False:
            print('Hello World!')
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BigBirdOnlyMLMHead(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.predictions = BigBirdLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BigBirdOnlyNSPHead(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        if False:
            i = 10
            return i + 15
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class BigBirdPreTrainingHeads(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.predictions = BigBirdLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        if False:
            i = 10
            return i + 15
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return (prediction_scores, seq_relationship_score)

class BigBirdPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BigBirdConfig
    load_tf_weights = load_tf_weights_in_big_bird
    base_model_prefix = 'bert'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights'
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
BIG_BIRD_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`BigBirdConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
BIG_BIRD_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@dataclass
class BigBirdForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BigBirdForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BigBirdForQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        pooler_output (`torch.FloatTensor` of shape `(batch_size, 1)`):
            pooler output from BigBigModel
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@add_start_docstrings('The bare BigBird Model transformer outputting raw hidden-states without any specific head on top.', BIG_BIRD_START_DOCSTRING)
class BigBirdModel(BigBirdPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.attention_type = self.config.attention_type
        self.config = config
        self.block_size = self.config.block_size
        self.embeddings = BigBirdEmbeddings(config)
        self.encoder = BigBirdEncoder(config)
        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation = nn.Tanh()
        else:
            self.pooler = None
            self.activation = None
        if self.attention_type != 'original_full' and config.add_cross_attention:
            logger.warning('When using `BigBirdForCausalLM` as decoder, then `attention_type` must be `original_full`. Setting `attention_type=original_full`')
            self.set_attention_type('original_full')
        self.post_init()

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.embeddings.word_embeddings = value

    def set_attention_type(self, value: str):
        if False:
            i = 10
            return i + 15
        if value not in ['original_full', 'block_sparse']:
            raise ValueError(f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}")
        if value == self.attention_type:
            return
        self.attention_type = value
        self.encoder.set_attention_type(value)

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, Tuple[torch.FloatTensor]]:
        if False:
            for i in range(10):
                print('nop')
        "\n        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if\n            the model is configured as a decoder.\n        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in\n            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        (batch_size, seq_length) = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        max_tokens_to_attend = (5 + 2 * self.config.num_random_blocks) * self.config.block_size
        if self.attention_type == 'block_sparse' and seq_length <= max_tokens_to_attend:
            sequence_length = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
            logger.warning(f"Attention type 'block_sparse' is not possible if sequence_length: {sequence_length} <= num global tokens: 2 * config.block_size + min. num sliding tokens: 3 * config.block_size + config.num_random_blocks * config.block_size + additional buffer: config.num_random_blocks * config.block_size = {max_tokens_to_attend} with config.block_size = {self.config.block_size}, config.num_random_blocks = {self.config.num_random_blocks}. Changing attention type to 'original_full'...")
            self.set_attention_type('original_full')
        if self.attention_type == 'block_sparse':
            (padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds) = self._pad_to_block_size(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, pad_token_id=self.config.pad_token_id)
        else:
            padding_len = 0
        if self.attention_type == 'block_sparse':
            (blocked_encoder_mask, band_mask, from_mask, to_mask) = self.create_masks_for_block_sparse_attn(attention_mask, self.block_size)
            extended_attention_mask = None
        elif self.attention_type == 'original_full':
            blocked_encoder_mask = None
            band_mask = None
            from_mask = None
            to_mask = None
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        else:
            raise ValueError(f'attention_type can either be original_full or block_sparse, but is {self.attention_type}')
        if self.config.is_decoder and encoder_hidden_states is not None:
            (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, band_mask=band_mask, from_mask=from_mask, to_mask=to_mask, blocked_encoder_mask=blocked_encoder_mask, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooler_output = self.activation(self.pooler(sequence_output[:, 0, :])) if self.pooler is not None else None
        if padding_len > 0:
            sequence_output = sequence_output[:, :-padding_len]
        if not return_dict:
            return (sequence_output, pooler_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooler_output, past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)

    @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask: torch.Tensor, block_size: int):
        if False:
            while True:
                i = 10
        (batch_size, seq_length) = attention_mask.size()
        if seq_length % block_size != 0:
            raise ValueError(f'Sequence length must be multiple of block size, but sequence length is {seq_length}, while block size is {block_size}.')

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            if False:
                print('Hello World!')
            '\n            Create 3D attention mask from a 2D tensor mask.\n\n            Args:\n                from_blocked_mask: 2D Tensor of shape [batch_size,\n                from_seq_length//from_block_size, from_block_size].\n                to_blocked_mask: int32 Tensor of shape [batch_size,\n                to_seq_length//to_block_size, to_block_size].\n\n            Returns:\n                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,\n                3*to_block_size].\n            '
            exp_blocked_to_pad = torch.cat([to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2)
            band_mask = torch.einsum('blq,blk->blqk', from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask.unsqueeze_(1)
            return band_mask
        blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)
        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)
        return (blocked_encoder_mask, band_mask, from_mask, to_mask)

    def _pad_to_block_size(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, position_ids: torch.Tensor, inputs_embeds: torch.Tensor, pad_token_id: int):
        if False:
            return 10
        'A helper function to pad tokens and mask to work with implementation of BigBird block-sparse attention.'
        block_size = self.config.block_size
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        (batch_size, seq_len) = input_shape[:2]
        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            logger.info(f'Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of `config.block_size`: {block_size}')
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full((batch_size, padding_len), self.config.pad_token_id, dtype=torch.long)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)
            attention_mask = nn.functional.pad(attention_mask, (0, padding_len), value=False)
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)
        return (padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds)

class BigBirdForPreTraining(BigBirdPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder.weight', 'cls.predictions.decoder.bias']

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.bert = BigBirdModel(config, add_pooling_layer=True)
        self.cls = BigBirdPreTrainingHeads(config)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        if False:
            i = 10
            return i + 15
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=BigBirdForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.FloatTensor]=None, next_sentence_label: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[BigBirdForPreTrainingOutput, Tuple[torch.FloatTensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the next sequence prediction (classification) loss. If specified, nsp loss will be\n            added to masked_lm loss. Input should be a sequence pair (see `input_ids` docstring) Indices should be in\n            `[0, 1]`:\n\n            - 0 indicates sequence B is a continuation of sequence A,\n            - 1 indicates sequence B is a random sequence.\n        kwargs (`Dict[str, any]`, optional, defaults to *{}*):\n            Used to hide legacy arguments that have been deprecated.\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, BigBirdForPreTraining\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")\n        >>> model = BigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base")\n\n        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")\n        >>> outputs = model(**inputs)\n\n        >>> prediction_logits = outputs.prediction_logits\n        >>> seq_relationship_logits = outputs.seq_relationship_logits\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        (sequence_output, pooled_output) = outputs[:2]
        (prediction_scores, seq_relationship_score) = self.cls(sequence_output, pooled_output)
        total_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if next_sentence_label is not None and total_loss is not None:
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = total_loss + next_sentence_loss
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return BigBirdForPreTrainingOutput(loss=total_loss, prediction_logits=prediction_scores, seq_relationship_logits=seq_relationship_score, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('BigBird Model with a `language modeling` head on top.', BIG_BIRD_START_DOCSTRING)
class BigBirdForMaskedLM(BigBirdPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder.weight', 'cls.predictions.decoder.bias']

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        if config.is_decoder:
            logger.warning('If you want to use `BigBirdForMaskedLM` make sure `config.is_decoder=False` for bi-directional self-attention.')
        self.bert = BigBirdModel(config)
        self.cls = BigBirdOnlyMLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            while True:
                i = 10
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[MaskedLMOutput, Tuple[torch.FloatTensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> import torch\n        >>> from transformers import AutoTokenizer, BigBirdForMaskedLM\n        >>> from datasets import load_dataset\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")\n        >>> model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base")\n        >>> squad_ds = load_dataset("squad_v2", split="train")  # doctest: +IGNORE_RESULT\n\n        >>> # select random long article\n        >>> LONG_ARTICLE_TARGET = squad_ds[81514]["context"]\n        >>> # select random sentence\n        >>> LONG_ARTICLE_TARGET[332:398]\n        \'the highest values are very close to the theoretical maximum value\'\n\n        >>> # add mask_token\n        >>> LONG_ARTICLE_TO_MASK = LONG_ARTICLE_TARGET.replace("maximum", "[MASK]")\n        >>> inputs = tokenizer(LONG_ARTICLE_TO_MASK, return_tensors="pt")\n        >>> # long article input\n        >>> list(inputs["input_ids"].shape)\n        [1, 919]\n\n        >>> with torch.no_grad():\n        ...     logits = model(**inputs).logits\n        >>> # retrieve index of [MASK]\n        >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]\n        >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)\n        >>> tokenizer.decode(predicted_token_id)\n        \'maximum\'\n        ```\n\n        ```python\n        >>> labels = tokenizer(LONG_ARTICLE_TARGET, return_tensors="pt")["input_ids"]\n        >>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)\n        >>> outputs = model(**inputs, labels=labels)\n        >>> round(outputs.loss.item(), 2)\n        1.99\n        ```\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        if False:
            return 10
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]
        if self.config.pad_token_id is None:
            raise ValueError('The PAD token should be defined for generation')
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full((effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([input_ids, dummy_token], dim=1)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

@add_start_docstrings('BigBird Model with a `language modeling` head on top for CLM fine-tuning.', BIG_BIRD_START_DOCSTRING)
class BigBirdForCausalLM(BigBirdPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder.weight', 'cls.predictions.decoder.bias']

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        if not config.is_decoder:
            logger.warning('If you want to use `BigBirdForCausalLM` as a standalone, add `is_decoder=True.`')
        self.bert = BigBirdModel(config)
        self.cls = BigBirdOnlyMLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            print('Hello World!')
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[CausalLMOutputWithCrossAttentions, Tuple[torch.FloatTensor]]:
        if False:
            while True:
                i = 10
        "\n        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if\n            the model is configured as a decoder.\n        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in\n            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in\n            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are\n            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        "
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        lm_loss = None
        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (lm_loss,) + output if lm_loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=lm_loss, logits=prediction_scores, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        if False:
            print('Hello World!')
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'past_key_values': past_key_values}

    def _reorder_cache(self, past_key_values, beam_idx):
        if False:
            print('Hello World!')
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past

class BigBirdClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, features, **kwargs):
        if False:
            print('Hello World!')
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@add_start_docstrings('\n    BigBird Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', BIG_BIRD_START_DOCSTRING)
class BigBirdForSequenceClassification(BigBirdPreTrainedModel):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BigBirdModel(config)
        self.classifier = BigBirdClassificationHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> import torch\n        >>> from transformers import AutoTokenizer, BigBirdForSequenceClassification\n        >>> from datasets import load_dataset\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("l-yohai/bigbird-roberta-base-mnli")\n        >>> model = BigBirdForSequenceClassification.from_pretrained("l-yohai/bigbird-roberta-base-mnli")\n        >>> squad_ds = load_dataset("squad_v2", split="train")  # doctest: +IGNORE_RESULT\n\n        >>> LONG_ARTICLE = squad_ds[81514]["context"]\n        >>> inputs = tokenizer(LONG_ARTICLE, return_tensors="pt")\n        >>> # long input article\n        >>> list(inputs["input_ids"].shape)\n        [1, 919]\n\n        >>> with torch.no_grad():\n        ...     logits = model(**inputs).logits\n        >>> predicted_class_id = logits.argmax().item()\n        >>> model.config.id2label[predicted_class_id]\n        \'LABEL_0\'\n        ```\n\n        ```python\n        >>> num_labels = len(model.config.id2label)\n        >>> model = BigBirdForSequenceClassification.from_pretrained(\n        ...     "l-yohai/bigbird-roberta-base-mnli", num_labels=num_labels\n        ... )\n        >>> labels = torch.tensor(1)\n        >>> loss = model(**inputs, labels=labels).loss\n        >>> round(loss.item(), 2)\n        1.13\n        ```\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    BigBird Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', BIG_BIRD_START_DOCSTRING)
class BigBirdForMultipleChoice(BigBirdPreTrainedModel):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.bert = BigBirdModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[MultipleChoiceModelOutput, Tuple[torch.FloatTensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,\n            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See\n            `input_ids` above)\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return MultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    BigBird Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', BIG_BIRD_START_DOCSTRING)
class BigBirdForTokenClassification(BigBirdPreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BigBirdModel(config)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[TokenClassifierOutput, Tuple[torch.FloatTensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class BigBirdForQuestionAnsweringHead(nn.Module):
    """Head for question answering tasks."""

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intermediate = BigBirdIntermediate(config)
        self.output = BigBirdOutput(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_output):
        if False:
            print('Hello World!')
        hidden_states = self.dropout(encoder_output)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states, encoder_output)
        hidden_states = self.qa_outputs(hidden_states)
        return hidden_states

@add_start_docstrings('\n    BigBird Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', BIG_BIRD_START_DOCSTRING)
class BigBirdForQuestionAnswering(BigBirdPreTrainedModel):

    def __init__(self, config, add_pooling_layer=False):
        if False:
            print('Hello World!')
        super().__init__(config)
        config.num_labels = 2
        self.num_labels = config.num_labels
        self.sep_token_id = config.sep_token_id
        self.bert = BigBirdModel(config, add_pooling_layer=add_pooling_layer)
        self.qa_classifier = BigBirdForQuestionAnsweringHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=BigBirdForQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, question_lengths: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[BigBirdForQuestionAnsweringModelOutput, Tuple[torch.FloatTensor]]:
        if False:
            while True:
                i = 10
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> import torch\n        >>> from transformers import AutoTokenizer, BigBirdForQuestionAnswering\n        >>> from datasets import load_dataset\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")\n        >>> model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base")\n        >>> squad_ds = load_dataset("squad_v2", split="train")  # doctest: +IGNORE_RESULT\n\n        >>> # select random article and question\n        >>> LONG_ARTICLE = squad_ds[81514]["context"]\n        >>> QUESTION = squad_ds[81514]["question"]\n        >>> QUESTION\n        \'During daytime how high can the temperatures reach?\'\n\n        >>> inputs = tokenizer(QUESTION, LONG_ARTICLE, return_tensors="pt")\n        >>> # long article and question input\n        >>> list(inputs["input_ids"].shape)\n        [1, 929]\n\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n\n        >>> answer_start_index = outputs.start_logits.argmax()\n        >>> answer_end_index = outputs.end_logits.argmax()\n        >>> predict_answer_token_ids = inputs.input_ids[0, answer_start_index : answer_end_index + 1]\n        >>> predict_answer_token = tokenizer.decode(predict_answer_token_ids)\n        ```\n\n        ```python\n        >>> target_start_index, target_end_index = torch.tensor([130]), torch.tensor([132])\n        >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)\n        >>> loss = outputs.loss\n        ```\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        seqlen = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
        if question_lengths is None and input_ids is not None:
            question_lengths = torch.argmax(input_ids.eq(self.sep_token_id).int(), dim=-1) + 1
            question_lengths.unsqueeze_(1)
        logits_mask = None
        if question_lengths is not None:
            logits_mask = self.prepare_question_mask(question_lengths, seqlen)
            if token_type_ids is None:
                token_type_ids = torch.ones(logits_mask.size(), dtype=int, device=logits_mask.device) - logits_mask
            logits_mask = logits_mask
            logits_mask[:, 0] = False
            logits_mask.unsqueeze_(2)
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.qa_classifier(sequence_output)
        if logits_mask is not None:
            logits = logits - logits_mask * 1000000.0
        (start_logits, end_logits) = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return BigBirdForQuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    @staticmethod
    def prepare_question_mask(q_lengths: torch.Tensor, maxlen: int):
        if False:
            for i in range(10):
                print('nop')
        mask = torch.arange(0, maxlen).to(q_lengths.device)
        mask.unsqueeze_(0)
        mask = torch.where(mask < q_lengths, 1, 0)
        return mask