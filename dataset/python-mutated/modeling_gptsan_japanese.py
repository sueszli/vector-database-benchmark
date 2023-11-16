""" PyTorch GPTSANJapanese model."""
import copy
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from ...activations import ACT2FN
from ...modeling_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings, add_start_docstrings_to_model_forward, is_torch_fx_proxy, logging
from .configuration_gptsan_japanese import GPTSanJapaneseConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'GPTSanJapaneseConfig'
_CHECKPOINT_FOR_DOC = 'Tanrei/GPTSAN-japanese'
GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST = ['Tanrei/GPTSAN-japanese']

def router_z_loss_func(router_logits: torch.Tensor) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the router z-loss implemented in PyTorch.\n\n    The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).\n    It encourages router logits to remain small in an effort to improve stability.\n\n    Args:\n        router_logits (`float`):\n            Input logits of shape [batch_size, sequence_length, num_experts]\n\n    Returns:\n        Scalar router z-loss.\n    '
    (num_groups, tokens_per_group, _) = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z ** 2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)

def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    if False:
        i = 10
        return i + 15
    '\n    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.\n\n    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss\n    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between\n    experts is too unbalanced.\n\n    Args:\n        router_probs (`torch.Tensor`):\n            Probability assigned to each expert per token. Shape: [batch_size, seqeunce_length, num_experts].\n        expert_indices (`torch.Tensor`):\n            Indices tensor of shape [batch_size, seqeunce_length] identifying the selected expert for a given token.\n\n    Returns:\n        The auxiliary loss.\n    '
    num_experts = router_probs.shape[-1]
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)
    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)
    expert_mask = torch.max(expert_mask, axis=-2).values
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)
    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * num_experts ** 2

class GPTSanJapaneseDenseActDense(nn.Module):
    """
    FFN Layer for Switch Transformer and Extra layers

    GPTSAN can mix Switch Transformer layers and normal Transformer layers This class is used as Expert in Switch
    Transformer layers and as FFN in regular Transformer layers. RELU is used in the Switch Transformer layer, and
    Swish is used in the normal Transformer layer, so there is a choice of which is used in the argument.

    """

    def __init__(self, config: GPTSanJapaneseConfig, ext_layer=False):
        if False:
            while True:
                i = 10
        super().__init__()
        d_inter = config.d_ext if ext_layer else config.d_ff
        self.wi = nn.Linear(config.d_model, d_inter, bias=ext_layer)
        self.wo = nn.Linear(d_inter, config.d_model, bias=ext_layer)
        self.dropout = nn.Identity() if ext_layer else nn.Dropout(config.dropout_rate)
        self.act = ACT2FN['swish' if ext_layer else 'relu']

    def forward(self, hidden_states):
        if False:
            print('Hello World!')
        '\n        Args:\n            hidden_states (`torch.Tensor`) :\n                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.\n        Returns:\n            torch.Tensor[num_groups, tokens_per_group, hidden_dim]\n\n        '
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class GPTSanJapaneseTop1Router(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: GPTSanJapaneseConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes router probabilities from input hidden states.\n\n        Args:\n            hidden_states (`torch.Tensor`):\n                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.\n        Returns:\n            router_probabilities (`torch.Tensor`):\n                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each\n                token and expert. Used for routing tokens to experts.\n            router_logits (`torch.Tensor`):\n                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.\n                This is used later for computing router z-loss.\n        '
        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)
        if self.jitter_noise > 0:
            distrib_lower_bound = 1.0 - self.jitter_noise
            distrib_upper_bound = 1.0 + self.jitter_noise
            uniform_distrib = torch.rand(hidden_states.shape, device=hidden_states.device, dtype=self.dtype)
            uniform_distrib = uniform_distrib * (distrib_lower_bound - distrib_upper_bound)
            uniform_distrib = uniform_distrib + distrib_upper_bound
            hidden_states *= uniform_distrib
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return (router_probabilities, router_logits)

    def _cast_classifier(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an\n        instance of the `Linear8bitLt` class by checking special attributes.\n        '
        if not (hasattr(self.classifier, 'SCB') or hasattr(self.classifier, 'CB')):
            self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        if False:
            print('Hello World!')
        '\n        Generic forward function for every Router class. Each Router expects to have the same input hidden states\n        (`hidden_states`) corresponding to the hidden states for each token, the `expert_capacity` corresponding to the\n        number of tokens the Router will send to each expert, some Routers can send up to few tokens to each expert.\n\n        Each Router works as the following: it expects the hidden states for each token, gets the `router_probs` and\n        `router_logits` from the `router_weights`. This will assign for each token, the raw probability to be assigned\n        to an expert. Then each Router class will have to define its own `_compute_routing_instructions`.\n\n        Args:\n            hidden_states (`torch.Tensor`) :\n                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.\n        Returns:\n            Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`] Tuple containing the expert index, the router probs\n            and the router logits. The router probabilities and logits are required to compute the loss.\n        '
        (router_probs, router_logits) = self._compute_router_probabilities(hidden_states)
        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)
        token_priority = torch.cumsum(expert_index, dim=-2)
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask
        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return (expert_index, router_probs, router_logits)

class GPTSanJapaneseSparseMLP(nn.Module):
    """
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: GPTSanJapaneseConfig, expert_class: nn.Module=GPTSanJapaneseDenseActDense):
        if False:
            return 10
        super().__init__()
        self.router = GPTSanJapaneseTop1Router(config)
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f'expert_{idx}'] = expert_class(config)

    def forward(self, hidden_states):
        if False:
            return 10
        '\n        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:\n\n        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`\n        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the\n        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).\n\n        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each\n        expert the corresponding hidden states.\n\n        '
        (router_mask, router_probs, router_logits) = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)
        next_states = hidden_states.clone()
        for (idx, expert) in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)
        hidden_states = router_probs * next_states
        return (hidden_states, (router_logits, expert_index))

class GPTSanJapaneseLayerSparseFF(nn.Module):
    """
    Switch Transformers Feed Forward layer module. This is a wrapper around the Mixture of Experts module.

    Parameters:
        config : ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(self, config: GPTSanJapaneseConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.mlp = GPTSanJapaneseSparseMLP(config)
        self.soft_bypass_mlp = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states, output_router_logits):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            hidden_states (`torch.Tensor`) :\n                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.\n            output_router_logits (`bool`) :\n                output experts router output.\n        Returns:\n            torch.Tensor[num_groups, tokens_per_group, hidden_dim]\n\n        '
        (forwarded_states, router_tuple) = self.mlp(hidden_states)
        forwarded_states += torch.tanh(self.soft_bypass_mlp(hidden_states))
        output = hidden_states + self.norm(forwarded_states)
        if output_router_logits and router_tuple is not None:
            return (output, router_tuple)
        else:
            return output

class GPTSanJapaneseLayerDenseFF(nn.Module):
    """
    Extra Transformers Feed Forward layer module.

    Parameters:
        config : ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(self, config: GPTSanJapaneseConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.mlp = GPTSanJapaneseDenseActDense(config, ext_layer=True)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`torch.Tensor`) :\n                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.\n        Returns:\n            torch.Tensor[num_groups, tokens_per_group, hidden_dim]\n\n        '
        forwarded_states = self.mlp(hidden_states)
        output = hidden_states + self.norm(forwarded_states)
        return output

class GPTSanJapaneseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, is_causal: bool=False, config: Optional[GPTSanJapaneseConfig]=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            return 10
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            print('Hello World!')
        'Input shape: Batch x Time x Channel'
        is_cross_attention = key_value_states is not None
        (bsz, tgt_len, _) = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None and (past_key_value[0].shape[2] == key_value_states.shape[1]):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}')
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped, past_key_value)

class GPTSanJapaneseLayerSelfAttention(nn.Module):
    """
    Self Attention and Normalization Unit
    """

    def __init__(self, config, has_relative_attention_bias=False):
        if False:
            print('Hello World!')
        super().__init__()
        self.self_attn = GPTSanJapaneseAttention(embed_dim=config.d_model, num_heads=config.num_heads, is_decoder=True, bias=has_relative_attention_bias)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if False:
            print('Hello World!')
        "\n        Self-attention and normalize block.\n\n        Args:\n            hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n                if the model is configured as a decoder.\n            past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up\n                decoding. If `past_key_values` are used, the user can optionally input only the last\n                `decoder_input_ids` (those that don't have their past key value states given to this model) of shape\n                `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used\n                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n            head_mask (`numpy.ndarray` of shape `({0})`, `optional):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            use_cache (`bool`, *optional*):\n                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding\n                (see `past_key_values`).\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        Returns:\n            Tuple[torch.Tensor[num_groups, tokens_per_group, hidden_dim],...]\n        "
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        atten_out = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=(1 - attention_mask) * torch.finfo(hidden_states.dtype).min, layer_head_mask=head_mask, output_attentions=output_attentions)
        if output_attentions:
            attn_weights = (atten_out[1],)
        else:
            attn_weights = ()
        attention_output = atten_out[0]
        hidden = hidden_states + self.norm(attention_output)
        if use_cache:
            outputs = (hidden, atten_out[2])
        else:
            outputs = (hidden,)
        return outputs + attn_weights

class GPTSanJapaneseBlock(nn.Module):
    """
    Self Attention and FFN Unit
    """

    def __init__(self, config, ext_layer=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.self_attn = GPTSanJapaneseLayerSelfAttention(config)
        self.feed_forward = GPTSanJapaneseLayerDenseFF(config) if ext_layer else GPTSanJapaneseLayerSparseFF(config)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False, output_router_tuple: Optional[bool]=False) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if False:
            print('Hello World!')
        "\n        GPTSAN transformer block.\n\n        Args:\n            hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n                if the model is configured as a decoder.\n            past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up\n                decoding. If `past_key_values` are used, the user can optionally input only the last\n                `decoder_input_ids` (those that don't have their past key value states given to this model) of shape\n                `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used\n                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n            head_mask (`numpy.ndarray` of shape `({0})`, `optional):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            use_cache (`bool`, *optional*):\n                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding\n                (see `past_key_values`).\n            output_attentions (`bool`) :\n                output attention probabirities.\n            output_router_tuple:\n                output experts router logits and expert id.\n        Returns:\n            Tuple[torch.Tensor[num_groups, tokens_per_group, hidden_dim],...]\n        "
        atten_out = self.self_attn(hidden_states=hidden_states, past_key_value=past_key_value, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attention_output = atten_out[0]
        if isinstance(self.feed_forward, GPTSanJapaneseLayerSparseFF):
            sparse_out = self.feed_forward(attention_output, output_router_tuple)
            if output_router_tuple:
                (hidden, router_tuple) = sparse_out
            else:
                hidden = sparse_out
        else:
            hidden = self.feed_forward(attention_output)
        outputs = (hidden,) + atten_out[1:]
        if isinstance(self.feed_forward, GPTSanJapaneseLayerSparseFF) and output_router_tuple:
            outputs += (router_tuple,)
        return outputs

class GPTSanJapanesePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTSanJapaneseConfig
    base_model_prefix = 'gptsan_japanese'
    supports_gradient_checkpointing = False
    _no_split_modules = ['GPTSanJapaneseBlock']
    _skip_keys_device_placement = 'past_key_values'

    @property
    def dummy_inputs(self):
        if False:
            while True:
                i = 10
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {'input_ids': input_ids, 'attention_mask': input_mask}
        return dummy_inputs

    def _init_weights(self, module):
        if False:
            print('Hello World!')
        'Initialize the weights'
        factor = self.config.initializer_factor
        if isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(factor * 1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, GPTSanJapaneseModel):
            module.embed_tokens.weight.data.normal_(mean=0.0, std=factor * 1.0)
            module.position_embeddings.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'extra_position_embeddings') and module.extra_position_embeddings is not None:
                module.extra_position_embeddings.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, (GPTSanJapaneseModel, GPTSanJapaneseForConditionalGeneration)):
            module.final_logits_bias.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'lm_head') and (not self.config.tie_word_embeddings):
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, GPTSanJapaneseDenseActDense):
            module.wi.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi, 'bias') and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, GPTSanJapaneseAttention):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_model
            n_heads = self.config.num_heads
            module.k_proj.weight.data.normal_(mean=0.0, std=factor * (d_model * key_value_proj_dim) ** (-0.5))
            module.v_proj.weight.data.normal_(mean=0.0, std=factor * (d_model * key_value_proj_dim) ** (-0.5))
            module.q_proj.weight.data.normal_(mean=0.0, std=factor * (d_model * key_value_proj_dim) ** (-0.5))
            module.out_proj.weight.data.normal_(mean=0.0, std=factor * (n_heads * key_value_proj_dim) ** (-0.5))
        elif isinstance(module, GPTSanJapaneseSparseMLP):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_model
            n_heads = self.config.num_heads
            module.router.classifier.weight.data.normal_(mean=0.0, std=factor * 1)
            for idx in range(self.config.num_experts):
                module.experts[f'expert_{idx}'].wi.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))
                module.experts[f'expert_{idx}'].wo.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))

    def _shift_right(self, input_ids):
        if False:
            for i in range(10):
                print('nop')
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        if decoder_start_token_id is None:
            raise ValueError('self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information.')
        if is_torch_fx_proxy(input_ids):
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
        if pad_token_id is None:
            raise ValueError('self.model.config.pad_token_id has to be defined.')
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids
GPTSAN_JAPANESE_START_DOCSTRING = '\n\n    The [GPTSAN-japanese](https://github.com/tanreinama/GPTSAN) model was proposed in General-purpose Swich transformer\n    based Japanese language model\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
GPTSAN_JAPANESE_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. GPTSAN-japanese is a model that generates sentence\n            continuations or predicts tokens at mask positions. Special tokens required for inputs to the model are\n            automatically appended.\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            An input that masks the Prefix part in the Prefix-LM input. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **prefix** input,\n            - 0 for tokens that are **not-prefix** input.\n        spout (`torch.Tensor` of shape `(batch_size, config.d_spout)`):\n                This vector is transformed through an 8-layer FFN and can be used instead of `past_key_values`.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded\n            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be\n            input (see `past_key_values`). This is useful if you want more control over how to convert\n            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`):\n            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.\n            Router logits of the decoder model, useful to compute the auxiliary loss for Mixture of Experts models.\n"

@add_start_docstrings('The bare GPTSAN-japanese Model transformer outputting raw hidden-states without any specific head on top.', GPTSAN_JAPANESE_START_DOCSTRING)
class GPTSanJapaneseModel(GPTSanJapanesePreTrainedModel):

    def __init__(self, config: GPTSanJapaneseConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.config = copy.deepcopy(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.last_project = nn.Linear(config.d_model, config.d_model, bias=True)
        self.act = ACT2FN['swish']
        self.blocks = torch.nn.ModuleList([])
        for _ in range(config.num_switch_layers):
            self.blocks.append(GPTSanJapaneseBlock(config))
        for _ in range(config.num_ext_layers):
            self.blocks.append(GPTSanJapaneseBlock(config, ext_layer=True))
        if config.num_ext_layers > 0:
            self.extra_position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        if config.d_spout:
            spouts = []
            for _ in range(8):
                spouts.append(nn.Linear(config.d_spout, config.d_spout, bias=False))
                spouts.append(nn.Tanh())
            spouts.append(nn.Linear(config.d_spout, config.num_layers * 2 * config.d_model, bias=False))
            self.spout = nn.Sequential(*spouts)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        if False:
            i = 10
            return i + 15
        self.embed_tokens = new_embeddings

    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.FloatTensor]=None, spout: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, head_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, output_router_logits: Optional[bool]=None, num_precontext: Optional[torch.LongTensor]=None) -> Union[MoEModelOutputWithPastAndCrossAttentions, Tuple[torch.FloatTensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        num_precontext (`torch.LongTensor` of shape `(batch_size,1)`):\n            length of `hybrid` input tokens in the input. Tokens up to this length refer to both front and back like\n            BERT, tokens after that refer only to front like GPT. see also:\n            https://github.com/tanreinama/GPTSAN/blob/main/report/model.md\n\n        Returns:\n            `MoEModelOutputWithPastAndCrossAttentions` or `tuple` if `return_dict` returns\n            MoEModelOutputWithPastAndCrossAttentions insted of tuple\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = self.position_embeddings.weight.device
        if input_ids is None:
            input_ids = torch.zeros([1, 1]).int().to(device)
        num_pasts_contexts = 0
        num_batch = input_ids.shape[0]
        pasts_or_spout_value = None
        if past_key_values is not None:
            num_pasts_contexts = past_key_values[0][0].shape[2]
        elif self.config.d_spout and spout is not None:
            num_pasts_contexts += 1
        if self.config.d_spout and spout is not None and (attention_mask is not None):
            attention_mask_with_spout = torch.ones(num_batch, attention_mask.shape[1] + 1, device=device)
            attention_mask_with_spout[:, 1:] -= 1 - attention_mask
            attention_mask = attention_mask_with_spout
        if num_precontext is not None:
            if not (len(num_precontext.shape) == 2 and num_precontext.shape[1] == 1):
                raise ValueError('num_precontext should be [batch, 1] size.')
            num_precontext = torch.reshape(num_precontext, [-1])
        else:
            num_precontext = torch.zeros([num_batch]).int().to(device)
        num_input_contexts = input_ids.shape[1]
        num_output_contexts = num_input_contexts + num_pasts_contexts
        hidden_states = self.embed_tokens(input_ids)
        if past_key_values is not None:
            pasts_or_spout_value = past_key_values
        elif self.config.d_spout and spout is not None:
            pasts_or_spout_value = self.spout(spout)
            pasts_or_spout_value = torch.reshape(pasts_or_spout_value, [num_batch, self.config.num_layers, 2, self.config.num_heads, num_pasts_contexts, self.config.d_model // self.config.num_heads])
            pasts_or_spout_value = torch.split(pasts_or_spout_value, [1] * self.config.num_layers, dim=1)
            pasts_or_spout_value = tuple((tuple([b.squeeze(1) for b in torch.split(a.squeeze(1), [1, 1], dim=1)]) for a in pasts_or_spout_value))
        else:
            pasts_or_spout_value = [None] * self.config.num_layers
        token_position = torch.arange(num_input_contexts).to(device) + num_pasts_contexts
        if attention_mask is None:
            attention_mask = torch.ones(num_batch, num_input_contexts, device=device)
        gather_position = (torch.zeros((num_batch, self.config.d_model, num_input_contexts)).to(device) + token_position.unsqueeze(0)).transpose(1, 2).long()
        gather_position -= (1 - attention_mask).argmin(dim=-1).unsqueeze(1).unsqueeze(2)
        gather_position = torch.clip(gather_position, num_pasts_contexts, self.config.max_position_embeddings - 1)
        for i in range(num_batch):
            hidden_states[i] += torch.gather(self.position_embeddings.weight, dim=0, index=gather_position[i])
        causal_mask = torch.tril(torch.ones((num_output_contexts, num_output_contexts), dtype=torch.uint8)).view(1, 1, num_output_contexts, num_output_contexts).to(device)
        prefix_lm_mask = causal_mask[:, :, -num_input_contexts:, :]
        if token_type_ids is not None:
            token_type_ids = token_type_ids.unsqueeze(1).unsqueeze(2)
            prefix_lm_mask = (prefix_lm_mask + token_type_ids > 0).float()
        extended_attention_mask = prefix_lm_mask * attention_mask.unsqueeze(1).unsqueeze(2)
        if head_mask is not None:
            head_mask = self.get_head_mask(head_mask, self.config.num_switch_layers + self.config.num_ext_layers)
        present_key_value_states = () if self.config.use_cache or use_cache else None
        all_hidden_states = () if self.config.output_hidden_states or output_hidden_states else None
        all_attentions = () if self.config.output_attentions or output_attentions else None
        all_router_probs = () if self.config.output_router_logits or output_router_logits else None
        for (layer, past) in enumerate(pasts_or_spout_value):
            if layer == self.config.num_switch_layers:
                if self.config.num_ext_layers > 0:
                    for i in range(num_batch):
                        hidden_states[i] += torch.gather(self.extra_position_embeddings.weight, dim=0, index=gather_position[i])
            output_router_tuple = (self.config.output_router_logits or output_router_logits) and layer < self.config.num_switch_layers
            block_output = self.blocks[layer](hidden_states=hidden_states, past_key_value=past, attention_mask=extended_attention_mask, head_mask=head_mask, use_cache=self.config.use_cache or use_cache, output_attentions=self.config.output_attentions or output_attentions, output_router_tuple=output_router_tuple)
            outpos = 0
            hidden_states = block_output[outpos]
            if self.config.output_hidden_states or output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.config.use_cache or use_cache:
                outpos += 1
                present = block_output[outpos]
                present_key_value_states += (present,)
            if self.config.output_attentions or output_attentions:
                outpos += 1
                attention_probs = block_output[outpos]
                all_attentions += (attention_probs,)
            if output_router_tuple:
                outpos += 1
                router_tuple = block_output[outpos]
                all_router_probs.append(router_tuple[0])
        hidden_states = self.last_project(hidden_states)
        hidden_states = self.act(hidden_states)
        if self.config.output_hidden_states or output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, present_key_value_states, all_hidden_states, all_attentions, all_router_probs] if v is not None))
        return MoEModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=present_key_value_states, hidden_states=all_hidden_states, attentions=all_attentions, router_probs=all_router_probs)

@add_start_docstrings('The bare GPTSAN-japanese Model with a language modeling head.', GPTSAN_JAPANESE_START_DOCSTRING)
class GPTSanJapaneseForConditionalGeneration(GPTSanJapanesePreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: GPTSanJapaneseConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.model = GPTSanJapaneseModel(config)
        self.register_buffer('final_logits_bias', torch.zeros([1, config.vocab_size]))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if not self.config.torchscript:
            self.lm_head.weight = self.model.embed_tokens.weight

    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.FloatTensor]=None, spout: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, head_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, output_router_logits: Optional[bool]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple[torch.FloatTensor], MoECausalLMOutputWithPast]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for\n            labels in `[0, ..., config.vocab_size]`\n\n        Returns:\n            `MoECausalLMOutputWithPast` or `tuple` if `return_dict` returns MoECausalLMOutputWithPast insted of tuple\n\n        Example:\n\n        Text Generation with regular LM Model\n        ```python\n        >>> from transformers import AutoModel, AutoTokenizer, trainer_utils\n\n        >>> device = "cuda"\n        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)\n        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")\n        >>> x_token = tokenizer("織田信長は、", return_tensors="pt")\n        >>> trainer_utils.set_seed(30)\n        >>> input_ids = x_token.input_ids.to(device)\n        >>> gen_token = model.generate(input_ids, max_new_tokens=50)\n        >>> tokenizer.decode(gen_token[0])\n        "織田信長は、政治・軍事の中枢まで掌握した政治家であり、日本史上類を見ない驚異的な軍事侵攻を続け..."\n        ```\n\n        Text Generation with Prefix-LM Model\n        ```python\n        >>> from transformers import AutoModel, AutoTokenizer, trainer_utils\n\n        >>> device = "cuda"\n        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)\n        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")\n        >>> x_token = tokenizer("", prefix_text="織田信長は、", return_tensors="pt")\n        >>> trainer_utils.set_seed(30)\n        >>> input_ids = x_token.input_ids.to(device)\n        >>> token_type_ids = x_token.token_type_ids.to(device)\n        >>> gen_token = model.generate(input_ids, token_type_ids=token_type_ids, max_new_tokens=50)\n        >>> tokenizer.decode(gen_token[0])\n        "織田信長は、政治・外交で数々の戦果を上げるが、1568年からは、いわゆる本能寺の変で細川晴元に暗殺される..."\n        ```\n\n        Simultaneously Text Generation And Masked Language Model\n        ```python\n        >>> from transformers import AutoModel, AutoTokenizer, trainer_utils\n\n        >>> device = "cuda"\n        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)\n        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")\n        >>> masked_sentence = "武田信玄は、<|inputmask|>時代ファンならぜひ押さえ<|inputmask|>きたい名将の一人。"\n        >>> x_token = tokenizer("", prefix_text=masked_sentence, return_tensors="pt")\n        >>> trainer_utils.set_seed(30)\n        >>> input_ids = x_token.input_ids.to(device)\n        >>> token_type_ids = x_token.token_type_ids.to(device)\n        >>> out_lm_token = model.generate(input_ids, token_type_ids=token_type_ids, max_new_tokens=50)\n        >>> out_mlm_token = model(input_ids, token_type_ids=token_type_ids).logits.argmax(axis=-1)\n        >>> tokenizer.decode(out_mlm_token[0])\n        "武田信玄は、戦国時代ファンならぜひ押さえておきたい名将の一人。"\n\n        >>> tokenizer.decode(out_lm_token[0][input_ids.shape[1] :])\n        "武田氏の三代に渡った武田家のひとり\\n甲斐市に住む、日本史上最大の戦国大名。..."\n        ```'
        SEG_TOKEN = self.config.separator_token_id
        use_cache = use_cache or self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_return_dict = True
        num_precontext = None
        if input_ids is not None:
            num_batch = input_ids.shape[0]
            num_precontext = torch.zeros([num_batch]).int().to(input_ids.device)
            where_separators = torch.where(input_ids == SEG_TOKEN)
            num_precontext[where_separators[0]] += where_separators[1]
            num_precontext = num_precontext.unsqueeze(1)
        outputs = self.model(input_ids, attention_mask, token_type_ids, spout, past_key_values, head_mask, use_cache, inputs_embeds, decoder_inputs_embeds, output_attentions, output_hidden_states, model_return_dict, output_router_logits, num_precontext)
        lm_logits = self.lm_head(outputs[0])
        if lm_logits.shape[-1] == self.final_logits_bias.shape[-1]:
            lm_logits = lm_logits + self.final_logits_bias
        loss = None
        z_loss = None
        router_probs = None
        aux_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            if output_router_logits:
                (router_logits, expert_indexes) = self._unpack_router_logits(outputs.router_probs)
                z_loss = router_z_loss_func(router_logits)
                router_probs = nn.Softmax(dim=-1)(router_logits)
                aux_loss = load_balancing_loss_func(router_probs, expert_indexes)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        if not return_dict:
            return tuple((v for v in [loss, lm_logits, outputs.past_key_values, outputs.hidden_states, outputs.router_probs, z_loss, aux_loss] if v is not None))
        return MoECausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, router_logits=outputs.router_probs, z_loss=z_loss, aux_loss=aux_loss)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor, token_type_ids: Optional[torch.FloatTensor]=None, spout: Optional[Union[List, torch.FloatTensor]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, **kwargs):
        if False:
            return 10
        if type(spout) is list:
            spout = torch.tensor(spout).float()
            if input_ids is not None:
                spout = spout.to(input_ids.device)
        if past_key_values is not None:
            return {'input_ids': input_ids[:, -1:] if input_ids is not None else None, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids[:, -1:] if token_type_ids is not None else None, 'spout': spout, 'past_key_values': past_key_values}
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'spout': spout, 'past_key_values': None}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        if False:
            return 10
        return self._shift_right(labels)

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int]=None) -> nn.Embedding:
        if False:
            return 10
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        if False:
            while True:
                i = 10
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer('final_logits_bias', new_bias)

    def get_input_embeddings(self):
        if False:
            return 10
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        if False:
            for i in range(10):
                print('nop')
        self.model.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.lm_head

    def _unpack_router_logits(self, router_outputs):
        if False:
            while True:
                i = 10
        total_router_logits = []
        total_expert_indexes = []
        for router_output in router_outputs:
            if len(router_output[0].shape) > 1:
                (router_logits, expert_indexes) = router_output
                total_router_logits.append(router_logits)
                total_expert_indexes.append(expert_indexes)
        return (torch.cat(total_router_logits, dim=1), torch.cat(total_expert_indexes, dim=1))