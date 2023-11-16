import logging
import torch
from torch import nn
from torch.nn import functional as F
from ludwig.utils.torch_utils import get_activation, LudwigModule
logger = logging.getLogger(__name__)

class FeedForwardAttentionReducer(LudwigModule):

    def __init__(self, input_size, hidden_size=256, activation='tanh'):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fc_layer1 = nn.Linear(input_size, hidden_size)
        self.fc_layer1_activation = get_activation(activation)
        self.fc_layer2 = nn.Linear(hidden_size, 1, bias=False)
        self.input_shape_var = None
        self.output_shape_var = None

    def forward(self, inputs, mask=None):
        if False:
            print('Hello World!')
        self.input_shape_var = inputs.size()[1:]
        hidden = self.fc_layer1(inputs)
        hidden = self.fc_layer1_activation(hidden)
        hidden = self.fc_layer2(hidden)
        attention = F.softmax(hidden, dim=1)
        gated_inputs = torch.sum(attention * inputs, dim=1)
        self.output_shape_var = gated_inputs.size()[1:]
        return gated_inputs

    @property
    def input_shape(self) -> torch.Size:
        if False:
            return 10
        return self.input_shape_var

    @property
    def output_shape(self) -> torch.Size:
        if False:
            return 10
        return self.output_shape_var

class MultiHeadSelfAttention(LudwigModule):

    def __init__(self, input_size, hidden_size, num_heads=8):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.embedding_size = hidden_size
        self.num_heads = num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(f'When using multi-head attention, `hidden_size` ({hidden_size}), should be divisible by `num_heads` ({num_heads}). Please update the `transformer` section of the model config.')
        self.projection_dim = hidden_size // num_heads
        self.query_dense = nn.Linear(input_size, hidden_size)
        self.key_dense = nn.Linear(input_size, hidden_size)
        self.value_dense = nn.Linear(input_size, hidden_size)
        self.combine_heads = nn.Linear(hidden_size, hidden_size)

    def attention(self, query, key, value, mask=None):
        if False:
            i = 10
            return i + 15
        score = torch.matmul(query, key.permute(0, 1, 3, 2))
        dim_key = torch.tensor(key.shape[-1]).type(torch.float32)
        scaled_score = score / torch.sqrt(dim_key)
        if mask:
            scaled_score = mask * scaled_score
        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return (output, weights)

    def separate_heads(self, inputs, batch_size):
        if False:
            for i in range(10):
                print('nop')
        inputs = torch.reshape(inputs, (batch_size, -1, self.num_heads, self.projection_dim))
        return torch.permute(inputs, (0, 2, 1, 3))

    def forward(self, inputs: torch.Tensor, mask=None):
        if False:
            while True:
                i = 10
        batch_size = inputs.shape[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        (outputs, weights) = self.attention(query, key, value, mask=mask)
        outputs = torch.permute(outputs, (0, 2, 1, 3))
        concat_outputs = torch.reshape(outputs, (batch_size, -1, self.embedding_size))
        projected_outputs = self.combine_heads(concat_outputs)
        return projected_outputs

    @property
    def output_shape(self):
        if False:
            i = 10
            return i + 15
        return torch.Size([self.embedding_size])

class TransformerBlock(LudwigModule):

    def __init__(self, input_size: int, max_sequence_length: int, hidden_size: int, num_heads: int, output_size: int, dropout: float=0.1):
        if False:
            while True:
                i = 10
        super().__init__()
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.self_attention = MultiHeadSelfAttention(input_size, hidden_size, num_heads=num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(hidden_size, eps=1e-06)
        self.fully_connected = nn.Sequential(nn.Linear(input_size, output_size), get_activation('relu'), nn.Linear(output_size, hidden_size))
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(hidden_size, eps=1e-06)

    @property
    def input_shape(self) -> torch.Size:
        if False:
            for i in range(10):
                print('nop')
        return torch.Size([self.max_sequence_length, self.input_size])

    def forward(self, inputs, mask=None):
        if False:
            while True:
                i = 10
        attn_output = self.self_attention(inputs)
        attn_output = self.dropout1(attn_output)
        ln1_output = self.layernorm1(inputs + attn_output)
        fc_output = self.fully_connected(ln1_output)
        fc_output = self.dropout2(fc_output)
        return self.layernorm2(ln1_output + fc_output)

    @property
    def output_shape(self) -> torch.Size:
        if False:
            print('Hello World!')
        return torch.Size([self.max_sequence_length, self.hidden_size])

class TransformerStack(LudwigModule):

    def __init__(self, input_size: int, max_sequence_length: int, hidden_size: int=256, num_heads: int=8, output_size: int=256, num_layers: int=1, dropout: float=0.1, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        self.supports_masking = True
        self.max_sequence_length = max_sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        prior_input_size = input_size
        for i in range(num_layers):
            layer = TransformerBlock(input_size=prior_input_size, max_sequence_length=max_sequence_length, hidden_size=hidden_size, num_heads=num_heads, output_size=output_size, dropout=dropout)
            self.layers.append(layer)
            prior_input_size = self.layers[i].output_shape[-1]
        for layer in self.layers:
            logger.debug(f'   {layer._get_name()}')

    @property
    def input_shape(self) -> torch.Size:
        if False:
            i = 10
            return i + 15
        return torch.Size([self.max_sequence_length, self.input_size])

    def forward(self, inputs, mask=None):
        if False:
            while True:
                i = 10
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden, mask=mask)
        return hidden

    @property
    def output_shape(self) -> torch.Size:
        if False:
            while True:
                i = 10
        return torch.Size([self.max_sequence_length, self.hidden_size])