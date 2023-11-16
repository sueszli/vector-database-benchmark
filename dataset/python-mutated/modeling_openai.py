"""PyTorch OpenAI GPT model."""
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import gelu_new, silu
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_openai import OpenAIGPTConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'openai-gpt'
_CONFIG_FOR_DOC = 'OpenAIGPTConfig'
OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = ['openai-gpt']

def load_tf_weights_in_openai_gpt(model, config, openai_checkpoint_folder_path):
    if False:
        i = 10
        return i + 15
    'Load tf pre-trained weights in a pytorch model (from NumPy arrays here)'
    import re
    import numpy as np
    if '.ckpt' in openai_checkpoint_folder_path:
        openai_checkpoint_folder_path = os.path.dirname(openai_checkpoint_folder_path)
    logger.info(f'Loading weights from {openai_checkpoint_folder_path}')
    with open(openai_checkpoint_folder_path + '/parameters_names.json', 'r', encoding='utf-8') as names_handle:
        names = json.load(names_handle)
    with open(openai_checkpoint_folder_path + '/params_shapes.json', 'r', encoding='utf-8') as shapes_handle:
        shapes = json.load(shapes_handle)
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(openai_checkpoint_folder_path + f'/params_{n}.npy') for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for (param, shape) in zip(init_params, shapes)]
    init_params = [arr.squeeze() for arr in init_params]
    if model.tokens_embed.weight.shape != init_params[1].shape:
        raise ValueError(f'tokens_embed.weight.shape: {model.tokens_embed.weight.shape} does not match init_param[1].shape: {init_params[1].shape}')
    if model.positions_embed.weight.shape != init_params[0].shape:
        raise ValueError(f'positions_embed.weight.shape: {model.positions_embed.weight.shape} does not match init_param[0].shape: {init_params[0].shape}')
    model.tokens_embed.weight.data = torch.from_numpy(init_params[1])
    model.positions_embed.weight.data = torch.from_numpy(init_params[0])
    names.pop(0)
    init_params.pop(0)
    init_params.pop(0)
    for (name, array) in zip(names, init_params):
        name = name[6:]
        if name[-2:] != ':0':
            raise ValueError(f'Layer {name} does not end with :0')
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+\\d+', m_name):
                scope_names = re.split('(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'g':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'b':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'w':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if pointer.shape != array.shape:
            raise ValueError(f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched')
        logger.info(f'Initialize PyTorch weight {name}')
        pointer.data = torch.from_numpy(array)
    return model
ACT_FNS = {'relu': nn.ReLU(), 'silu': silu, 'gelu': gelu_new, 'swish': silu}

class Attention(nn.Module):

    def __init__(self, nx, n_positions, config, scale=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        n_state = nx
        if n_state % config.n_head != 0:
            raise ValueError(f'Attention n_state shape: {n_state} must be divisible by config.n_head {config.n_head}')
        self.register_buffer('bias', torch.tril(torch.ones(n_positions, n_positions)).view(1, 1, n_positions, n_positions), persistent=False)
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            while True:
                i = 10
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.n_head, self.split_size // self.n_head, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + 2 * self.split_size])
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        self.split_size = self.split_size // self.n_head * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        if False:
            print('Hello World!')
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        b = self.bias[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -10000.0 * (1 - b)
        if attention_mask is not None:
            w = w + attention_mask
        w = nn.functional.softmax(w, dim=-1)
        w = self.attn_dropout(w)
        if head_mask is not None:
            w = w * head_mask
        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        if False:
            while True:
                i = 10
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        if False:
            print('Hello World!')
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        if False:
            for i in range(10):
                print('nop')
        x = self.c_attn(x)
        (query, key, value) = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = [a] + attn_outputs[1:]
        return outputs

class MLP(nn.Module):

    def __init__(self, n_state, config):
        if False:
            return 10
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        if False:
            print('Hello World!')
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class Block(nn.Module):

    def __init__(self, n_positions, config, scale=False):
        if False:
            while True:
                i = 10
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_positions, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        if False:
            return 10
        attn_outputs = self.attn(x, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions)
        a = attn_outputs[0]
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        outputs = [h] + attn_outputs[1:]
        return outputs

class OpenAIGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = OpenAIGPTConfig
    load_tf_weights = load_tf_weights_in_openai_gpt
    base_model_prefix = 'transformer'

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the weights.'
        if isinstance(module, (nn.Linear, Conv1D)):
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

@dataclass
class OpenAIGPTDoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        mc_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mc_labels` is provided):
            Multiple choice classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
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
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
OPENAI_GPT_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`OpenAIGPTConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
OPENAI_GPT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top.', OPENAI_GPT_START_DOCSTRING)
class OpenAIGPTModel(OpenAIGPTPreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_positions, config, scale=True) for _ in range(config.n_layer)])
        self.register_buffer('position_ids', torch.arange(config.n_positions), persistent=False)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.tokens_embed

    def set_input_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.tokens_embed = new_embeddings

    def _prune_heads(self, heads_to_prune):
        if False:
            while True:
                i = 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}\n        '
        for (layer, heads) in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        if False:
            print('Hello World!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if position_ids is None:
            position_ids = self.position_ids[None, :input_shape[-1]]
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if inputs_embeds is None:
            inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for (i, block) in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(hidden_states, attention_mask, head_mask[i], output_attentions=output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)
        hidden_states = hidden_states.view(*output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

@add_start_docstrings('\n    OpenAI GPT Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', OPENAI_GPT_START_DOCSTRING)
class OpenAIGPTLMHeadModel(OpenAIGPTPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            return 10
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], CausalLMOutput]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set\n            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`\n            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutput(loss=loss, logits=lm_logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {'input_ids': input_ids}

@add_start_docstrings('\nOpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for\nRocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the\ninput embeddings, the classification head takes as input the input of a specified classification token index in the\ninput sequence).\n', OPENAI_GPT_START_DOCSTRING)
class OpenAIGPTDoubleHeadsModel(OpenAIGPTPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        config.num_labels = 1
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OpenAIGPTDoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, mc_token_ids: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, mc_labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], OpenAIGPTDoubleHeadsModelOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices)`, *optional*, default to index of the last token of the input):\n            Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) -\n            1]`.\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set\n            `labels = input_ids` Indices are selected in `[-1, 0, ..., config.vocab_size]` All labels set to `-100` are\n            ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`\n        mc_labels (`torch.LongTensor` of shape `(batch_size)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`\n            where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)\n\n        Return:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, OpenAIGPTDoubleHeadsModel\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")\n        >>> model = OpenAIGPTDoubleHeadsModel.from_pretrained("openai-gpt")\n        >>> tokenizer.add_special_tokens(\n        ...     {"cls_token": "[CLS]"}\n        ... )  # Add a [CLS] to the vocabulary (we should train it also!)\n        >>> model.resize_token_embeddings(len(tokenizer))\n\n        >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]\n        >>> input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices\n        >>> mc_token_ids = torch.tensor([input_ids.size(-1) - 1, input_ids.size(-1) - 1]).unsqueeze(0)  # Batch size 1\n\n        >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)\n        >>> lm_logits = outputs.logits\n        >>> mc_logits = outputs.mc_logits\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
        (lm_loss, mc_loss) = (None, None)
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return (lm_loss,) + output if lm_loss is not None else output
        return OpenAIGPTDoubleHeadsModelOutput(loss=lm_loss, mc_loss=mc_loss, logits=lm_logits, mc_logits=mc_logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    The Original OpenAI GPT Model transformer with a sequence classification head on top (linear layer).\n    [`OpenAIGPTForSequenceClassification`] uses the last token in order to do the classification, as other causal\n    models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the\n    last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding\n    token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since\n    it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take\n    the last value in each row of the batch).\n    ', OPENAI_GPT_START_DOCSTRING)
class OpenAIGPTForSequenceClassification(OpenAIGPTPreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = OpenAIGPTModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
        self.post_init()

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        if input_ids is not None:
            (batch_size, sequence_length) = input_ids.shape[:2]
        else:
            (batch_size, sequence_length) = inputs_embeds.shape[:2]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError('Cannot handle batch sizes > 1 if no padding token is defined.')
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(logits.device)
        else:
            sequence_lengths = -1
            logger.warning(f'{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be unexpected if using padding tokens in conjunction with `inputs_embeds.`')
        pooled_logits = logits[range(batch_size), sequence_lengths]
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
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=pooled_logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)