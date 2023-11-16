import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from TTS.tts.layers.delightful_tts.conv_layers import Conv1dGLU, DepthWiseConv1d, PointwiseConv1d
from TTS.tts.layers.delightful_tts.networks import GLUActivation

def calc_same_padding(kernel_size: int) -> Tuple[int, int]:
    if False:
        print('Hello World!')
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

class Conformer(nn.Module):

    def __init__(self, dim: int, n_layers: int, n_heads: int, speaker_embedding_dim: int, p_dropout: float, kernel_size_conv_mod: int, lrelu_slope: float):
        if False:
            for i in range(10):
                print('nop')
        '\n        A Transformer variant that integrates both CNNs and Transformers components.\n        Conformer proposes a novel combination of self-attention and convolution, in which self-attention\n        learns the global interaction while the convolutions efficiently capture the local correlations.\n\n        Args:\n            dim (int): Number of the dimensions for the model.\n            n_layers (int): Number of model layers.\n            n_heads (int): The number of attention heads.\n            speaker_embedding_dim (int): Number of speaker embedding dimensions.\n            p_dropout (float): Probabilty of dropout.\n            kernel_size_conv_mod (int): Size of kernels for convolution modules.\n\n        Inputs: inputs, mask\n            - **inputs** (batch, time, dim): Tensor containing input vector\n            - **encoding** (batch, time, dim): Positional embedding tensor\n            - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked\n        Returns:\n            - **outputs** (batch, time, dim): Tensor produced by Conformer Encoder.\n        '
        super().__init__()
        d_k = d_v = dim // n_heads
        self.layer_stack = nn.ModuleList([ConformerBlock(dim, n_heads, d_k, d_v, kernel_size_conv_mod=kernel_size_conv_mod, dropout=p_dropout, speaker_embedding_dim=speaker_embedding_dim, lrelu_slope=lrelu_slope) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor, speaker_embedding: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Shapes:\n            - x: :math:`[B, T_src, C]`\n            - mask: :math: `[B]`\n            - speaker_embedding: :math: `[B, C]`\n            - encoding: :math: `[B, T_max2, C]`\n        '
        attn_mask = mask.view((mask.shape[0], 1, 1, mask.shape[1]))
        for enc_layer in self.layer_stack:
            x = enc_layer(x, mask=mask, slf_attn_mask=attn_mask, speaker_embedding=speaker_embedding, encoding=encoding)
        return x

class ConformerBlock(torch.nn.Module):

    def __init__(self, d_model: int, n_head: int, d_k: int, d_v: int, kernel_size_conv_mod: int, speaker_embedding_dim: int, dropout: float, lrelu_slope: float=0.3):
        if False:
            i = 10
            return i + 15
        '\n        A Conformer block is composed of four modules stacked together,\n        A feed-forward module, a self-attention module, a convolution module,\n        and a second feed-forward module in the end. The block starts with two Feed forward\n        modules sandwiching the Multi-Headed Self-Attention module and the Conv module.\n\n        Args:\n            d_model (int): The dimension of model\n            n_head (int): The number of attention heads.\n            kernel_size_conv_mod (int): Size of kernels for convolution modules.\n            speaker_embedding_dim (int): Number of speaker embedding dimensions.\n            emotion_embedding_dim (int): Number of emotion embedding dimensions.\n            dropout (float): Probabilty of dropout.\n\n        Inputs: inputs, mask\n            - **inputs** (batch, time, dim): Tensor containing input vector\n            - **encoding** (batch, time, dim): Positional embedding tensor\n            - **slf_attn_mask** (batch, 1, 1, time1): Tensor containing indices to be masked in self attention module\n            - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked\n        Returns:\n            - **outputs** (batch, time, dim): Tensor produced by the Conformer Block.\n        '
        super().__init__()
        if isinstance(speaker_embedding_dim, int):
            self.conditioning = Conv1dGLU(d_model=d_model, kernel_size=kernel_size_conv_mod, padding=kernel_size_conv_mod // 2, embedding_dim=speaker_embedding_dim)
        self.ff = FeedForward(d_model=d_model, dropout=dropout, kernel_size=3, lrelu_slope=lrelu_slope)
        self.conformer_conv_1 = ConformerConvModule(d_model, kernel_size=kernel_size_conv_mod, dropout=dropout, lrelu_slope=lrelu_slope)
        self.ln = nn.LayerNorm(d_model)
        self.slf_attn = ConformerMultiHeadedSelfAttention(d_model=d_model, num_heads=n_head, dropout_p=dropout)
        self.conformer_conv_2 = ConformerConvModule(d_model, kernel_size=kernel_size_conv_mod, dropout=dropout, lrelu_slope=lrelu_slope)

    def forward(self, x: torch.Tensor, speaker_embedding: torch.Tensor, mask: torch.Tensor, slf_attn_mask: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Shapes:\n            - x: :math:`[B, T_src, C]`\n            - mask: :math: `[B]`\n            - slf_attn_mask: :math: `[B, 1, 1, T_src]`\n            - speaker_embedding: :math: `[B, C]`\n            - emotion_embedding: :math: `[B, C]`\n            - encoding: :math: `[B, T_max2, C]`\n        '
        if speaker_embedding is not None:
            x = self.conditioning(x, embeddings=speaker_embedding)
        x = self.ff(x) + x
        x = self.conformer_conv_1(x) + x
        res = x
        x = self.ln(x)
        (x, _) = self.slf_attn(query=x, key=x, value=x, mask=slf_attn_mask, encoding=encoding)
        x = x + res
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = self.conformer_conv_2(x) + x
        return x

class FeedForward(nn.Module):

    def __init__(self, d_model: int, kernel_size: int, dropout: float, lrelu_slope: float, expansion_factor: int=4):
        if False:
            while True:
                i = 10
        '\n        Feed Forward module for conformer block.\n\n        Args:\n            d_model (int): The dimension of model.\n            kernel_size (int): Size of the kernels for conv layers.\n            dropout (float): probability of dropout.\n            expansion_factor (int): The factor by which to project the number of channels.\n            lrelu_slope (int): the negative slope factor for the leaky relu activation.\n\n        Inputs: inputs\n            - **inputs** (batch, time, dim): Tensor containing input vector\n        Returns:\n            - **outputs** (batch, time, dim): Tensor produced by the feed forward module.\n        '
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.conv_1 = nn.Conv1d(d_model, d_model * expansion_factor, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.LeakyReLU(lrelu_slope)
        self.conv_2 = nn.Conv1d(d_model * expansion_factor, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Shapes:\n            x: :math: `[B, T, C]`\n        '
        x = self.ln(x)
        x = x.permute((0, 2, 1))
        x = self.conv_1(x)
        x = x.permute((0, 2, 1))
        x = self.act(x)
        x = self.dropout(x)
        x = x.permute((0, 2, 1))
        x = self.conv_2(x)
        x = x.permute((0, 2, 1))
        x = self.dropout(x)
        x = 0.5 * x
        return x

class ConformerConvModule(nn.Module):

    def __init__(self, d_model: int, expansion_factor: int=2, kernel_size: int=7, dropout: float=0.1, lrelu_slope: float=0.3):
        if False:
            i = 10
            return i + 15
        '\n        Convolution module for conformer. Starts with a gating machanism.\n        a pointwise convolution and a gated linear unit (GLU). This is followed\n        by a single 1-D depthwise convolution layer. Batchnorm is deployed just after the convolution\n        to help with training. it also contains an expansion factor to project the number of channels.\n\n        Args:\n            d_model (int): The dimension of model.\n            expansion_factor (int): The factor by which to project the number of channels.\n            kernel_size (int): Size of kernels for convolution modules.\n            dropout (float): Probabilty of dropout.\n            lrelu_slope (float): The slope coefficient for leaky relu activation.\n\n        Inputs: inputs\n            - **inputs** (batch, time, dim): Tensor containing input vector\n        Returns:\n            - **outputs** (batch, time, dim): Tensor produced by the conv module.\n\n        '
        super().__init__()
        inner_dim = d_model * expansion_factor
        self.ln_1 = nn.LayerNorm(d_model)
        self.conv_1 = PointwiseConv1d(d_model, inner_dim * 2)
        self.conv_act = GLUActivation(slope=lrelu_slope)
        self.depthwise = DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=calc_same_padding(kernel_size)[0])
        self.ln_2 = nn.GroupNorm(1, inner_dim)
        self.activation = nn.LeakyReLU(lrelu_slope)
        self.conv_2 = PointwiseConv1d(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Shapes:\n            x: :math: `[B, T, C]`\n        '
        x = self.ln_1(x)
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = self.conv_act(x)
        x = self.depthwise(x)
        x = self.ln_2(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x

class ConformerMultiHeadedSelfAttention(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float):
        if False:
            return 10
        super().__init__()
        self.attention = RelativeMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            return 10
        (batch_size, seq_length, _) = key.size()
        encoding = encoding[:, :key.shape[1]]
        encoding = encoding.repeat(batch_size, 1, 1)
        (outputs, attn) = self.attention(query, key, value, pos_embedding=encoding, mask=mask)
        outputs = self.dropout(outputs)
        return (outputs, attn)

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(self, d_model: int=512, num_heads: int=16):
        if False:
            print('Hello World!')
        super().__init__()
        assert d_model % num_heads == 0, 'd_model % num_heads should be zero.'
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pos_embedding: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            i = 10
            return i + 15
        batch_size = query.shape[0]
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        u_bias = self.u_bias.expand_as(query)
        v_bias = self.v_bias.expand_as(query)
        a = (query + u_bias).transpose(1, 2)
        content_score = a @ key.transpose(2, 3)
        b = (query + v_bias).transpose(1, 2)
        pos_score = b @ pos_embedding.permute(0, 2, 3, 1)
        pos_score = self._relative_shift(pos_score)
        score = content_score + pos_score
        score = score * (1.0 / self.sqrt_dim)
        score.masked_fill_(mask, -1000000000.0)
        attn = F.softmax(score, -1)
        context = (attn @ value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)
        return (self.out_proj(context), attn)

    def _relative_shift(self, pos_score: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        (batch_size, num_heads, seq_length1, seq_length2) = pos_score.size()
        zeros = torch.zeros((batch_size, num_heads, seq_length1, 1), device=pos_score.device)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)
        return pos_score

class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim: int, key_dim: int, num_units: int, num_heads: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        querys = self.W_query(query)
        keys = self.W_key(key)
        values = self.W_value(key)
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)
        scores = torch.matmul(querys, keys.transpose(2, 3))
        scores = scores / self.key_dim ** 0.5
        scores = F.softmax(scores, dim=3)
        out = torch.matmul(scores, values)
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)
        return out