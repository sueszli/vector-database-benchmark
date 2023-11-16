"""

Generic interface to various configurations of the Perceiver Resampler, that simply takes in a series of (potentially
time-indexed) contextual embeddings, and "resamples" (compresses) them down to a pre-specified number of latents! Note
that the Perceiver in general resamples based solely off the *long-range* context; there's a nice opportunity here to
prime the Perceiver Resampler with say a single layer's worth of language embeddings (the target domain), and use that
to softly "retrieve & compress" what we need --> this would be a novel contribution we should explore.

References:
    - DeepMind's Flamingo: https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model
    - Code borrowed w/ love from: https://github.com/lucidrains/flamingo-pytorch

"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .configuration_idefics import IdeficsConfig

class IdeficsPerceiverResampler(nn.Module):

    def __init__(self, config: IdeficsConfig, embed_dim: int, depth: int, n_heads: int, head_dim: int, n_latents: int) -> None:
        if False:
            while True:
                i = 10
        '\n        Instantiates a Perceiver Resampler that operates over a sequence of embeddings (say from a ResNet or ViT or\n        MAE) of a given dimension, performs `depth` blocks of cross-attention with a fixed `n_latents` inputs, then\n        returns a Tensor of shape [bsz, n_latents, embed_dim]. :param embed_dim: Dimensionality of embeddings being fed\n        to the Perceiver Resampler (also dimensionality of latent embeddings *returned* by the Perceiver Resampler.\n        Could be e.g., VIT embed_dim, ResNet pool dim, and so on.\n\n        Args:\n            config (`IdeficsConfig`): config object\n            embed_dim (`int`): The size of each embedding vector\n            depth (`int`): Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (< 3).\n            n_heads (`int`): Number of heads in each Transformer block (for multi-headed self-attention).\n            head_dim (`int`): Dimensionality of each head projection in the Transformer block.\n            n_latents (`int`):\n                Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).\n\n        '
        super().__init__()
        (self.embed_dim, self.n_heads, self.head_dim, self.n_latents) = (embed_dim, n_heads, head_dim, n_latents)
        self.qk_layer_norms = config.perceiver_config.qk_layer_norms_perceiver
        self.latents = nn.Parameter(torch.randn(self.n_latents, self.embed_dim), requires_grad=True)
        self.intermediate_dim = self.embed_dim * 4 if not hasattr(config.vision_config, 'embed_dim') else config.vision_config.embed_dim * 4
        self.blocks = nn.ModuleList([nn.ModuleList([IdeficsPerceiverAttention(self.embed_dim, self.n_heads, self.head_dim, self.qk_layer_norms), IdeficsMLP(self.intermediate_dim, config)]) for _ in range(depth)])
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        'Resample arbitrary length context & *compress* down to self.n_latents latent embeddings'
        latents = self.latents.repeat(context.shape[0], 1, 1)
        for (attn, ff) in self.blocks:
            latents = attn(context, latents) + latents
            latents = ff(latents) + latents
        return self.layer_norm(latents)

class IdeficsPerceiverAttention(nn.Module):

    def __init__(self, embed_dim: int, n_heads: int, head_dim: int, qk_layer_norms: bool) -> None:
        if False:
            while True:
                i = 10
        'Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`'
        super().__init__()
        (self.embed_dim, self.n_heads, self.head_dim) = (embed_dim, n_heads, head_dim)
        self.qk_layer_norms = qk_layer_norms
        self.context_layer_norm = nn.LayerNorm(self.embed_dim)
        self.latents_layer_norm = nn.LayerNorm(self.embed_dim)
        if self.qk_layer_norms:
            self.q_layer_norm = nn.LayerNorm(self.head_dim)
            self.k_layer_norm = nn.LayerNorm(self.head_dim)
        self.qk_scale = self.head_dim ** (-0.5)
        self.q_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(self.n_heads * self.head_dim, embed_dim, bias=False)

    def forward(self, context: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!\n\n        Args:\n            context (`torch.Tensor`):\n                Tensor of shape `[bsz, seq, embed_dim]` representing long-form context to resample.\n            latents (`torch.Tensor`):\n                Tensor of shape `[bsz, n_latents, embed_dim]` representing fixed length latents to compress to.\n\n        Returns:\n            `torch.Tensor`: Tensor of shape `[bsz, n_latents, embed_dim]` representing attention over latents w/ cross\n            from context.\n        '
        context = self.context_layer_norm(context)
        latents = self.latents_layer_norm(latents)
        (batch_size, seq_length, embed_dim) = context.shape[:3]
        q = self.q_proj(latents)
        k = self.k_proj(torch.cat([context, latents], dim=-2))
        v = self.v_proj(torch.cat([context, latents], dim=-2))
        (q, k, v) = [x.reshape(batch_size, x.shape[1], self.n_heads, self.head_dim).transpose(1, 2) for x in (q, k, v)]
        if self.qk_layer_norms:
            q = self.q_layer_norm(q)
            k = self.k_layer_norm(k)
        scores = torch.einsum('... i d, ... j d -> ... i j', q * self.qk_scale, k)
        stabilized_scores = scores - scores.amax(dim=-1, keepdim=True).detach()
        attn = stabilized_scores.softmax(dim=-1)
        resampled = torch.einsum('... i j, ... j d -> ... i d', attn, v)
        return self.output_proj(resampled.transpose(1, 2).flatten(-2))

class IdeficsMLP(nn.Module):

    def __init__(self, intermediate_size, config: IdeficsConfig):
        if False:
            return 10
        'Simple MLP block with intermediate_size and embedding size'
        super().__init__()
        self.embed_dim = config.vision_config.embed_dim
        self.ln = nn.LayerNorm(self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, intermediate_size, bias=False)
        self.act = nn.ReLU()
        self.c_proj = nn.Linear(intermediate_size, self.embed_dim, bias=False)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.ln(hidden_states)
        hidden_states = self.fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states