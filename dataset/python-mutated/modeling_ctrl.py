""" PyTorch CTRL model."""
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_ctrl import CTRLConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'CTRLConfig'
CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = ['Salesforce/ctrl']

def angle_defn(pos, i, d_model_size):
    if False:
        while True:
            i = 10
    angle_rates = 1 / torch.pow(10000, 2 * (i // 2) / d_model_size)
    return pos * angle_rates

def positional_encoding(position, d_model_size, dtype):
    if False:
        for i in range(10):
            print('nop')
    angle_rads = angle_defn(torch.arange(position, dtype=dtype).unsqueeze(1), torch.arange(d_model_size, dtype=dtype).unsqueeze(0), d_model_size)
    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])
    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding

def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):
    if False:
        while True:
            i = 10
    matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))
    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    if mask is not None:
        (nd, ns) = (scaled_attention_logits.size(-2), scaled_attention_logits.size(-1))
        scaled_attention_logits += mask[ns - nd:ns, :ns] * -10000.0
    if attention_mask is not None:
        scaled_attention_logits = scaled_attention_logits + attention_mask
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    if head_mask is not None:
        attention_weights = attention_weights * head_mask
    output = torch.matmul(attention_weights, v)
    return (output, attention_weights)

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model_size, num_heads):
        if False:
            while True:
                i = 10
        super().__init__()
        self.num_heads = num_heads
        self.d_model_size = d_model_size
        self.depth = int(d_model_size / self.num_heads)
        self.Wq = nn.Linear(d_model_size, d_model_size)
        self.Wk = nn.Linear(d_model_size, d_model_size)
        self.Wv = nn.Linear(d_model_size, d_model_size)
        self.dense = nn.Linear(d_model_size, d_model_size)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            print('Hello World!')
        attention_head_size = self.d_model_size // self.num_heads
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.num_heads, attention_head_size, self.pruned_heads)
        self.Wq = prune_linear_layer(self.Wq, index)
        self.Wk = prune_linear_layer(self.Wk, index)
        self.Wv = prune_linear_layer(self.Wv, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)
        self.num_heads = self.num_heads - len(heads)
        self.d_model_size = attention_head_size * self.num_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def split_into_heads(self, x, batch_size):
        if False:
            while True:
                i = 10
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute([0, 2, 1, 3])

    def forward(self, v, k, q, mask, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        if False:
            for i in range(10):
                print('nop')
        batch_size = q.shape[0]
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        if layer_past is not None:
            (past_key, past_value) = (layer_past[0], layer_past[1])
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        if use_cache is True:
            present = torch.stack((k, v))
        else:
            present = (None,)
        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)
        scaled_attention = output[0].permute([0, 2, 1, 3])
        attn = output[1]
        original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
        output = self.dense(original_size_attention)
        outputs = (output, present)
        if output_attentions:
            outputs = outputs + (attn,)
        return outputs

def point_wise_feed_forward_network(d_model_size, dff):
    if False:
        i = 10
        return i + 15
    return nn.Sequential(nn.Linear(d_model_size, dff), nn.ReLU(), nn.Linear(dff, d_model_size))

class EncoderLayer(nn.Module):

    def __init__(self, d_model_size, num_heads, dff, rate=0.1):
        if False:
            print('Hello World!')
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model_size, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model_size, dff)
        self.layernorm1 = nn.LayerNorm(d_model_size, eps=1e-06)
        self.layernorm2 = nn.LayerNorm(d_model_size, eps=1e-06)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        if False:
            while True:
                i = 10
        normed = self.layernorm1(x)
        attn_outputs = self.multi_head_attention(normed, normed, normed, mask, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output
        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output
        outputs = (out2,) + attn_outputs[1:]
        return outputs

class CTRLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CTRLConfig
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
CTRL_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`CTRLConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
CTRL_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0].shape[-2]`\n            (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.\n\n            If `past_key_values` is used, only input IDs that do not have their past calculated should be passed as\n            `input_ids`.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        past_key_values (`Tuple[Tuple[torch.FloatTensor]]` of length `config.n_layers`):\n            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see\n            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have\n            their past given to this model should not be passed as input ids as they have already been computed.\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.', CTRL_START_DOCSTRING)
class CTRLModel(CTRLPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer
        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size, torch.float)
        self.w = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([EncoderLayer(config.n_embd, config.n_head, config.dff, config.resid_pdrop) for _ in range(config.n_layer)])
        self.layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.w

    def set_input_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.w = new_embeddings

    def _prune_heads(self, heads_to_prune):
        if False:
            print('Hello World!')
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}\n        '
        for (layer, heads) in heads_to_prune.items():
            self.h[layer].multi_head_attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, CTRLModel\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")\n        >>> model = CTRLModel.from_pretrained("Salesforce/ctrl")\n\n        >>> # CTRL was trained with control codes as the first token\n        >>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")\n        >>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()\n\n        >>> outputs = model(**inputs)\n\n        >>> last_hidden_states = outputs.last_hidden_state\n        >>> list(last_hidden_states.shape)\n        [1, 5, 1280]\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError('batch_size has to be defined and > 0')
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
            token_type_embeds = self.w(token_type_ids)
            token_type_embeds *= np.sqrt(self.d_model_size)
        else:
            token_type_embeds = 0
        if inputs_embeds is None:
            inputs_embeds = self.w(input_ids)
        seq_len = input_shape[-1]
        mask = torch.triu(torch.ones(seq_len + past_length, seq_len + past_length), 1).to(device)
        inputs_embeds *= np.sqrt(self.d_model_size)
        self.pos_encoding = self.pos_encoding.to(device)
        pos_embeds = self.pos_encoding[position_ids, :]
        hidden_states = inputs_embeds + pos_embeds + token_type_embeds
        hidden_states = self.dropout(hidden_states)
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for (i, (h, layer_past)) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = h(hidden_states, mask, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions)
            (hidden_states, present) = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)
            if output_attentions:
                all_attentions += (outputs[2],)
        hidden_states = self.layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_attentions)

@add_start_docstrings('\n    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', CTRL_START_DOCSTRING)
class CTRLLMHeadModel(CTRLPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.transformer = CTRLModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            for i in range(10):
                print('nop')
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_cache=None, **kwargs):
        if False:
            print('Hello World!')
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        return {'input_ids': input_ids, 'past_key_values': past_key_values, 'use_cache': use_cache}

    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        if False:
            return 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set\n            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`\n            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> import torch\n        >>> from transformers import AutoTokenizer, CTRLLMHeadModel\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")\n        >>> model = CTRLLMHeadModel.from_pretrained("Salesforce/ctrl")\n\n        >>> # CTRL was trained with control codes as the first token\n        >>> inputs = tokenizer("Wikipedia The llama is", return_tensors="pt")\n        >>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()\n\n        >>> sequence_ids = model.generate(inputs["input_ids"])\n        >>> sequences = tokenizer.batch_decode(sequence_ids)\n        >>> sequences\n        [\'Wikipedia The llama is a member of the family Bovidae. It is native to the Andes of Peru,\']\n\n        >>> outputs = model(**inputs, labels=inputs["input_ids"])\n        >>> round(outputs.loss.item(), 2)\n        9.21\n\n        >>> list(outputs.logits.shape)\n        [1, 5, 246534]\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
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
        return CausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    @staticmethod
    def _reorder_cache(past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or\n        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct\n        beam_idx at every generation step.\n        '
        return tuple((tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)) for layer_past in past_key_values))

@add_start_docstrings('\n    The CTRL Model transformer with a sequence classification head on top (linear layer).\n    [`CTRLForSequenceClassification`] uses the last token in order to do the classification, as other causal models\n    (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the last\n    token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in\n    each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot\n    guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last\n    value in each row of the batch).\n    ', CTRL_START_DOCSTRING)
class CTRLForSequenceClassification(CTRLPreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = CTRLModel(config)
        self.classifier = nn.Linear(config.n_embd, self.num_labels, bias=False)
        self.post_init()

    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n\n        Example of single-label classification:\n\n        ```python\n        >>> import torch\n        >>> from transformers import AutoTokenizer, CTRLForSequenceClassification\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")\n        >>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl")\n\n        >>> # CTRL was trained with control codes as the first token\n        >>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")\n        >>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()\n\n        >>> with torch.no_grad():\n        ...     logits = model(**inputs).logits\n\n        >>> predicted_class_id = logits.argmax().item()\n        >>> model.config.id2label[predicted_class_id]\n        \'LABEL_0\'\n        ```\n\n        ```python\n        >>> import torch\n\n        >>> torch.manual_seed(42)  # doctest: +IGNORE_RESULT\n        >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`\n        >>> num_labels = len(model.config.id2label)\n        >>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)\n\n        >>> labels = torch.tensor(1)\n        >>> loss = model(**inputs, labels=labels).loss\n        >>> round(loss.item(), 2)\n        0.35\n        ```\n\n        Example of multi-label classification:\n\n        ```python\n        >>> import torch\n        >>> from transformers import AutoTokenizer, CTRLForSequenceClassification\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")\n        >>> model = CTRLForSequenceClassification.from_pretrained(\n        ...     "Salesforce/ctrl", problem_type="multi_label_classification"\n        ... )\n\n        >>> # CTRL was trained with control codes as the first token\n        >>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")\n        >>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()\n\n        >>> with torch.no_grad():\n        ...     logits = model(**inputs).logits\n\n        >>> predicted_class_id = logits.argmax().item()\n        >>> model.config.id2label[predicted_class_id]\n        \'LABEL_0\'\n        ```\n\n        ```python\n        >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`\n        >>> num_labels = len(model.config.id2label)\n        >>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)\n\n        >>> num_labels = len(model.config.id2label)\n        >>> labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(\n        ...     torch.float\n        ... )\n        >>> loss = model(**inputs, labels=labels).loss\n        >>> loss.backward()  # doctest: +IGNORE_RESULT\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        logits = self.classifier(hidden_states)
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
            output = (pooled_logits,) + transformer_outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=pooled_logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)