import torch.nn as nn
from .learned_positional_embedding import LearnedPositionalEmbedding
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

def PositionalEmbedding(num_embeddings: int, embedding_dim: int, padding_idx: int, learned: bool=False):
    if False:
        i = 10
        return i + 15
    if learned:
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** (-0.5))
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, init_size=num_embeddings + padding_idx + 1)
    return m