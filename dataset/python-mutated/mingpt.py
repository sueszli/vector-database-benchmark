"""
Adapted from https://github.com/karpathy/minGPT

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers
        /models/gpt2/modeling_gpt2.py
"""
import math
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated

@DeveloperAPI
@dataclass
class GPTConfig:
    block_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    embed_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

@Deprecated(error=False)
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT
    repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper:
    https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        if False:
            while True:
                i = 10
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

@Deprecated(error=False)
class CausalSelfAttention(nn.Module):
    """
    Vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config: GPTConfig):
        if False:
            print('Hello World!')
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x, attention_masks=None):
        if False:
            while True:
                i = 10
        (B, T, C) = x.size()
        (q, k, v) = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        if attention_masks is not None:
            att = att + attention_masks
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return (y, att)

@Deprecated(error=False)
class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config: GPTConfig):
        if False:
            return 10
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = nn.ModuleDict(dict(c_fc=nn.Linear(config.n_embed, 4 * config.n_embed), c_proj=nn.Linear(4 * config.n_embed, config.n_embed), act=NewGELU(), dropout=nn.Dropout(config.resid_pdrop)))

    def forward(self, x, attention_masks=None):
        if False:
            for i in range(10):
                print('nop')
        (x_att, att) = self.attn(self.ln_1(x), attention_masks=attention_masks)
        x = x + x_att
        x_ffn = self.mlp.dropout(self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(x))))
        x = x + x_ffn
        return (x, att)

@Deprecated(error=False)
def configure_gpt_optimizer(model: nn.Module, learning_rate: float, weight_decay: float, betas: Tuple[float, float]=(0.9, 0.95), **kwargs) -> torch.optim.Optimizer:
    if False:
        return 10
    "\n    This long function is unfortunately doing something very simple and is\n    being very defensive: We are separating out all parameters of the model\n    into two buckets: those that will experience weight decay for regularization\n    and those that won't (biases, and layernorm/embedding weights). We are then\n    returning the PyTorch optimizer object.\n    "
    decay = set()
    no_decay = set()
    whitelist_w_modules = (torch.nn.Linear,)
    blacklist_w_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for (mn, m) in model.named_modules():
        for (pn, p) in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_w_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_w_modules):
                no_decay.add(fpn)
    param_dict = {pn: p for (pn, p) in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f'parameters {str(inter_params)} made it into both decay/no_decay sets!'
    assert len(param_dict.keys() - union_params) == 0, f'parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!'
    optim_groups = [{'params': [param_dict[pn] for pn in sorted(decay)], 'weight_decay': weight_decay}, {'params': [param_dict[pn] for pn in sorted(no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **kwargs)
    return optimizer

@Deprecated(error=False)
class GPT(nn.Module):
    """GPT Transformer Model"""

    def __init__(self, config: GPTConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        assert config.block_size is not None
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(drop=nn.Dropout(config.embed_pdrop), h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), ln_f=nn.LayerNorm(config.n_embed)))
        self.apply(self._init_weights)
        for (pn, p) in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_embeds, attention_masks=None, return_attentions=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        input_embeds: [batch_size x seq_len x n_embed]\n        attention_masks: [batch_size x seq_len], 0 don't attend, 1 attend\n        "
        (B, T, C) = input_embeds.size()
        assert T <= self.block_size, f'Cannot forward sequence of length {T}, block size is only {self.block_size}'
        if attention_masks is not None:
            (_B, _T) = attention_masks.size()
            assert _B == B and _T == T
            attention_masks = attention_masks[:, None, None, :]
            attention_masks = attention_masks.to(dtype=input_embeds.dtype)
            attention_masks = (1.0 - attention_masks) * -1000000000.0
        x = self.transformer.drop(input_embeds)
        atts = []
        for block in self.transformer.h:
            (x, att) = block(x, attention_masks=attention_masks)
            atts.append(att)
        x = self.transformer.ln_f(x)
        if return_attentions:
            return (x, atts)
        else:
            return x