from typing import Sequence, Dict, List, Callable
import torch
import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.highway import Highway
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
_VALID_PROJECTION_LOCATIONS = {'after_cnn', 'after_highway', None}

@Seq2VecEncoder.register('cnn-highway')
class CnnHighwayEncoder(Seq2VecEncoder):
    """
    The character CNN + highway encoder from
    [Kim et al "Character aware neural language models"](https://arxiv.org/abs/1508.06615)
    with an optional projection.

    Registered as a `Seq2VecEncoder` with name "cnn-highway".

    # Parameters

    embedding_dim : `int`, required
        The dimension of the initial character embedding.
    filters : `Sequence[Sequence[int]]`, required
        A sequence of pairs (filter_width, num_filters).
    num_highway : `int`, required
        The number of highway layers.
    projection_dim : `int`, required
        The output dimension of the projection layer.
    activation : `str`, optional (default = `'relu'`)
        The activation function for the convolutional layers.
    projection_location : `str`, optional (default = `'after_highway'`)
        Where to apply the projection layer. Valid values are
        'after_highway', 'after_cnn', and None.
    """

    def __init__(self, embedding_dim: int, filters: Sequence[Sequence[int]], num_highway: int, projection_dim: int, activation: str='relu', projection_location: str='after_highway', do_layer_norm: bool=False) -> None:
        if False:
            return 10
        super().__init__()
        if projection_location not in _VALID_PROJECTION_LOCATIONS:
            raise ConfigurationError(f'unknown projection location: {projection_location}')
        self.input_dim = embedding_dim
        self.output_dim = projection_dim
        self._projection_location = projection_location
        if activation == 'tanh':
            self._activation = torch.nn.functional.tanh
        elif activation == 'relu':
            self._activation = torch.nn.functional.relu
        else:
            raise ConfigurationError(f'unknown activation {activation}')
        self._convolutions: List[torch.nn.Module] = []
        for (i, (width, num)) in enumerate(filters):
            conv = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=num, kernel_size=width, bias=True)
            conv.weight.data.uniform_(-0.05, 0.05)
            conv.bias.data.fill_(0.0)
            self.add_module(f'char_conv_{i}', conv)
            self._convolutions.append(conv)
        num_filters = sum((num for (_, num) in filters))
        if projection_location == 'after_cnn':
            highway_dim = projection_dim
        else:
            highway_dim = num_filters
        self._highways = Highway(highway_dim, num_highway, activation=torch.nn.functional.relu)
        for highway_layer in self._highways._layers:
            highway_layer.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / highway_dim))
            highway_layer.bias[:highway_dim].data.fill_(0.0)
            highway_layer.bias[highway_dim:].data.fill_(2.0)
        self._projection = torch.nn.Linear(num_filters, projection_dim, bias=True)
        self._projection.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / num_filters))
        self._projection.bias.data.fill_(0.0)
        if do_layer_norm:
            self._layer_norm: Callable = LayerNorm(self.output_dim)
        else:
            self._layer_norm = lambda tensor: tensor

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> Dict[str, torch.Tensor]:
        if False:
            return 10
        '\n        Compute context insensitive token embeddings for ELMo representations.\n\n        # Parameters\n\n        inputs: `torch.Tensor`\n            Shape `(batch_size, num_characters, embedding_dim)`\n            Character embeddings representing the current batch.\n        mask: `torch.BoolTensor`\n            Shape `(batch_size, num_characters)`\n            Currently unused. The mask for characters is implicit. See TokenCharactersEncoder.forward.\n\n        # Returns\n\n        `encoding`:\n            Shape `(batch_size, projection_dim)` tensor with context-insensitive token representations.\n        '
        inputs = inputs.transpose(1, 2)
        convolutions = []
        for i in range(len(self._convolutions)):
            char_conv_i = getattr(self, f'char_conv_{i}')
            convolved = char_conv_i(inputs)
            (convolved, _) = torch.max(convolved, dim=-1)
            convolved = self._activation(convolved)
            convolutions.append(convolved)
        token_embedding = torch.cat(convolutions, dim=-1)
        if self._projection_location == 'after_cnn':
            token_embedding = self._projection(token_embedding)
        token_embedding = self._highways(token_embedding)
        if self._projection_location == 'after_highway':
            token_embedding = self._projection(token_embedding)
        token_embedding = self._layer_norm(token_embedding)
        return token_embedding

    def get_input_dim(self) -> int:
        if False:
            print('Hello World!')
        return self.input_dim

    def get_output_dim(self) -> int:
        if False:
            print('Hello World!')
        return self.output_dim