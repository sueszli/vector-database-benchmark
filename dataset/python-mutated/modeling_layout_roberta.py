import math
import os
import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from transformers.activations import ACT2FN, gelu
from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, CausalLMOutputWithCrossAttentions, MaskedLMOutput, MultipleChoiceModelOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel, apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
logger = logging.get_logger()

class LayoutRobertaConfig(PretrainedConfig):
    model_type = 'layoutroberta'

    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=1, bos_token_id=0, eos_token_id=2, bbox_scale=100.0, pe_type='crel', position_embedding_type='absolute', use_cache=True, classifier_dropout=None, **kwargs):
        if False:
            return 10
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size, hidden_act=hidden_act, hidden_dropout_prob=hidden_dropout_prob, attention_probs_dropout_prob=attention_probs_dropout_prob, max_position_embeddings=max_position_embeddings, type_vocab_size=type_vocab_size, initializer_range=initializer_range, layer_norm_eps=layer_norm_eps, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.bbox_scale = bbox_scale
        self.pe_type = pe_type

class PositionalEmbedding1D(nn.Module):

    def __init__(self, demb):
        if False:
            for i in range(10):
                print('nop')
        super(PositionalEmbedding1D, self).__init__()
        self.demb = demb
        inv_freq = 1 / 10000 ** (torch.arange(0.0, demb, 2.0) / demb)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        if False:
            while True:
                i = 10
        seq_size = pos_seq.size()
        if len(seq_size) == 2:
            (b1, b2) = seq_size
            sinusoid_inp = pos_seq.view(b1, b2, 1) * self.inv_freq.view(1, 1, self.demb // 2)
        elif len(seq_size) == 3:
            (b1, b2, b3) = seq_size
            sinusoid_inp = pos_seq.view(b1, b2, b3, 1) * self.inv_freq.view(1, 1, 1, self.demb // 2)
        else:
            raise ValueError(f'Invalid seq_size={len(seq_size)}')
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb

class PositionalEmbedding2D(nn.Module):

    def __init__(self, demb, dim_bbox=8):
        if False:
            return 10
        super(PositionalEmbedding2D, self).__init__()
        self.demb = demb
        self.dim_bbox = dim_bbox
        self.x_pos_emb = PositionalEmbedding1D(demb // dim_bbox)
        self.y_pos_emb = PositionalEmbedding1D(demb // dim_bbox)
        inv_freq = 1 / 10000 ** (torch.arange(0.0, demb, 2.0) / demb)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, bbox):
        if False:
            return 10
        stack = []
        for i in range(self.dim_bbox):
            if i % 2 == 0:
                stack.append(self.x_pos_emb(bbox[..., i]))
            else:
                stack.append(self.y_pos_emb(bbox[..., i]))
        bbox_pos_emb = torch.cat(stack, dim=-1)
        return bbox_pos_emb

class LayoutRobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.line_rank_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.line_rank_inner_embeddings = nn.Embedding(4, config.hidden_size)
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse('1.6.0'):
            self.register_buffer('token_type_ids', torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)
        if config.pe_type == 'pdpdq_ws':
            dim_bbox_sinusoid_emb = config.hidden_size
            dim_bbox_projection = config.hidden_size
        elif config.pe_type == 'crel':
            dim_bbox_sinusoid_emb = config.hidden_size // 4
            dim_bbox_projection = config.hidden_size // config.num_attention_heads
        else:
            raise ValueError(f'Unknown config.pe_type={config.pe_type}')
        self.bbox_sinusoid_emb = PositionalEmbedding2D(dim_bbox_sinusoid_emb, dim_bbox=8)
        self.bbox_projection = nn.Linear(dim_bbox_sinusoid_emb, dim_bbox_projection, bias=False)
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)

    def _cal_spatial_position_embeddings(self, bbox):
        if False:
            return 10
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError('The :obj:`bbox`coordinate values should be within 0-1000 range.') from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        spatial_position_embeddings = torch.cat([left_position_embeddings, upper_position_embeddings, right_position_embeddings, lower_position_embeddings, h_position_embeddings, w_position_embeddings], dim=-1)
        return spatial_position_embeddings

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if 'line_bbox' in kwargs:
            embeddings += self._cal_spatial_position_embeddings(kwargs['line_bbox'])
        if 'line_rank_id' in kwargs:
            embeddings += self.line_rank_embeddings(kwargs['line_rank_id'])
        if 'line_rank_inner_id' in kwargs:
            embeddings += self.line_rank_inner_embeddings(kwargs['line_rank_inner_id'])
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def calc_bbox_pos_emb(self, bbox, pe_type):
        if False:
            for i in range(10):
                print('nop')
        bbox_t = bbox.transpose(0, 1)
        if pe_type == 'pdpdq_ws':
            bbox_pos = bbox_t
        elif pe_type == 'crel':
            bbox_pos = bbox_t[None, :, :, :] - bbox_t[:, None, :, :]
        else:
            raise ValueError(f'Unknown pe_type={pe_type}')
        bbox_pos_emb = self.bbox_sinusoid_emb(bbox_pos)
        bbox_pos_emb = self.bbox_projection(bbox_pos_emb)
        return bbox_pos_emb

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        if False:
            print('Hello World!')
        '\n        We are provided embeddings directly.\n        We cannot infer which are padded so just generate sequential position ids.\n\n        Args:\n            inputs_embeds: torch.Tensor\n\n        Returns: torch.Tensor\n        '
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape)

class LayoutRobertaSelfAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.is_decoder = config.is_decoder
        self.pe_type = config.pe_type

    def transpose_for_scores(self, x):
        if False:
            print('Hello World!')
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, bbox_pos_emb=None, bbox_pos_mask=None):
        if False:
            print('Hello World!')
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
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum('bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        (batch_size, n_head, seq_length, d_head) = query_layer.shape
        if self.pe_type == 'pdpdq_ws':
            head_q_pos = self.query(bbox_pos_emb)
            head_k_pos = self.key(bbox_pos_emb)
            head_q_pos = head_q_pos.view(seq_length, batch_size, n_head, d_head)
            head_k_pos = head_k_pos.view(seq_length, batch_size, n_head, d_head)
            head_q_pos = head_q_pos.permute([1, 2, 0, 3])
            head_k_pos = head_k_pos.permute([1, 2, 0, 3])
            bbox_pos_scores_1 = torch.einsum('bnid,bnjd->bnij', (torch.mul(query_layer, head_q_pos), head_k_pos))
            bbox_pos_scores_2 = torch.einsum('bnid,bnjd->bnij', (head_q_pos, head_k_pos))
            bbox_pos_scores = bbox_pos_scores_1 + bbox_pos_scores_2
        elif self.pe_type == 'crel':
            bbox_pos_emb = bbox_pos_emb.view(seq_length, seq_length, batch_size, d_head)
            bbox_pos_emb = bbox_pos_emb.permute([2, 0, 1, 3])
            bbox_pos_scores = torch.einsum('bnid,bijd->bnij', (query_layer, bbox_pos_emb))
        else:
            raise ValueError(f'Unknown self.pe_type={self.pe_type}')
        if bbox_pos_mask is not None:
            bbox_pos_mask = 1 - bbox_pos_mask
            M1 = bbox_pos_mask.unsqueeze(1)
            MT = M1.permute(0, 2, 1)
            bbox_pos_mask_final = torch.matmul(MT.to(bbox_pos_scores.dtype), M1.to(bbox_pos_scores.dtype))
        else:
            bbox_pos_mask_final = None
        if bbox_pos_mask_final is not None:
            bbox_pos_scores = torch.mul(bbox_pos_scores, bbox_pos_mask_final.unsqueeze(1))
        attention_scores = attention_scores + bbox_pos_scores
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
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class LayoutRobertaSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class LayoutRobertaAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.self = LayoutRobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = LayoutRobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            return 10
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, bbox_pos_emb=None, bbox_pos_mask=None):
        if False:
            print('Hello World!')
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, bbox_pos_emb=bbox_pos_emb, bbox_pos_mask=bbox_pos_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class LayoutRobertaIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class LayoutRobertaOutput(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class LayoutRobertaLayer(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutRobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = LayoutRobertaAttention(config, position_embedding_type='absolute')
        self.intermediate = LayoutRobertaIntermediate(config)
        self.output = LayoutRobertaOutput(config)

    def forward(self, hidden_states, attention_mask=None, bbox_pos_emb=None, bbox_pos_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        if False:
            return 10
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value, bbox_pos_emb=bbox_pos_emb, bbox_pos_mask=bbox_pos_mask)
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise ValueError(f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`')
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
            i = 10
            return i + 15
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class LayoutRobertaEncoder(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LayoutRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True, bbox_pos_emb=None, bbox_pos_mask=None):
        if False:
            i = 10
            return i + 15
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                    use_cache = False

                def create_custom_forward(module):
                    if False:
                        return 10

                    def custom_forward(*inputs):
                        if False:
                            return 10
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, bbox_pos_emb, bbox_pos_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, bbox_pos_emb, bbox_pos_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
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

class LayoutRobertaPooler(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        if False:
            return 10
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class LayoutRobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LayoutRobertaConfig
    base_model_prefix = 'layoutroberta'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = ['position_ids']

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
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

    def _set_gradient_checkpointing(self, module, value=False):
        if False:
            print('Hello World!')
        if isinstance(module, LayoutRobertaEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        if False:
            print('Hello World!')
        'Remove some keys from ignore list'
        if not config.tie_word_embeddings:
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore]

class LayoutRobertaModel(LayoutRobertaPreTrainedModel):
    """

    BROS + Roberta

    """

    def __init__(self, config, add_pooling_layer=True):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.config = config
        self.embeddings = LayoutRobertaEmbeddings(config)
        self.encoder = LayoutRobertaEncoder(config)
        self.pooler = LayoutRobertaPooler(config) if add_pooling_layer else None
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

    def _prune_heads(self, heads_to_prune):
        if False:
            return 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, bbox=None, bbox_mask=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if\n            the model is configured as a decoder.\n        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in\n            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple\n            having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        "
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
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        if self.config.is_decoder and encoder_hidden_states is not None:
            (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length, **kwargs)
        scaled_bbox = bbox * self.config.bbox_scale
        bbox_pos_emb = self.embeddings.calc_bbox_pos_emb(scaled_bbox, self.config.pe_type)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, bbox_pos_emb=bbox_pos_emb, bbox_pos_mask=bbox_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooled_output, past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    if False:
        i = 10
        return i + 15
    "\n    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols\n    are ignored. This is modified from fairseq's `utils.make_positions`.\n\n    Args:\n        x: torch.Tensor x:\n\n    Returns: torch.Tensor\n    "
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx