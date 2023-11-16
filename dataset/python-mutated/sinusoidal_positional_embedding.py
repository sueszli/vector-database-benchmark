import math
from typing import Any, Optional
import torch
import torch.onnx.operators
from fairseq import utils
from torch import nn, Tensor

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        if False:
            return 10
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.register_buffer('weights', SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim, padding_idx), persistent=False)
        self.max_positions = int(100000.0)
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        if False:
            print('Hello World!')
        self.onnx_trace = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if False:
            while True:
                i = 10
        deprecated_keys = ['weights', '_float_tensor']
        for key in deprecated_keys:
            if prefix + key in state_dict:
                del state_dict[prefix + key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        'Build sinusoidal embeddings.\n\n        This matches the implementation in tensor2tensor, but differs slightly\n        from the description in Section 3.5 of "Attention Is All You Need".\n        '
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state: Optional[Any]=None, timestep: Optional[Tensor]=None, positions: Optional[Any]=None):
        if False:
            for i in range(10):
                print('nop')
        'Input is expected to be of size [bsz x seqlen].'
        bspair = torch.onnx.operators.shape_as_tensor(input)
        (bsz, seq_len) = (bspair[0], bspair[1])
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(max_pos, self.embedding_dim, self.padding_idx).to(self.weights)
        if incremental_state is not None:
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(bsz, 1, 1)
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
        positions = utils.make_positions(input, self.padding_idx, onnx_trace=self.onnx_trace)
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat((bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long)))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(flat_embeddings, embedding_shape)
            return embeddings
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()