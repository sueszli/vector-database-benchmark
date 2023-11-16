import logging
import torch
from ludwig.modules.attention_modules import FeedForwardAttentionReducer
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import LudwigModule, sequence_length_3D
logger = logging.getLogger(__name__)

class SequenceReducer(LudwigModule):
    """Reduces the sequence dimension of an input tensor according to the specified reduce_mode.  Any additional
    kwargs are passed on to the reduce mode's constructor.  If using reduce_mode=="attention", the input_size kwarg
    must also be specified.

    A sequence is a tensor of 2 or more dimensions, where the shape is [batch size x sequence length x ...].

    :param reduce_mode: The reduction mode, one of {"last", "sum", "mean", "max", "concat", "attention", "none"}
    :param max_sequence_length The maximum sequence length.  Only used for computation of shapes - inputs passed
                               at runtime may have a smaller sequence length.
    :param encoding_size The size of each sequence element/embedding vector, or None if input is a sequence of scalars.
    """

    def __init__(self, reduce_mode: str=None, max_sequence_length: int=256, encoding_size: int=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._reduce_mode = reduce_mode
        self._max_sequence_length = max_sequence_length
        self._encoding_size = encoding_size
        if reduce_mode == 'attention' and encoding_size and ('input_size' not in kwargs):
            kwargs['input_size'] = encoding_size
        self._reduce_obj = get_from_registry(reduce_mode, reduce_mode_registry)(**kwargs)

    def forward(self, inputs, mask=None):
        if False:
            print('Hello World!')
        'Forward pass of reducer.\n\n        :param inputs: A tensor of 2 or more dimensions, where the shape is [batch size x sequence length x ...].\n        :param mask: A mask tensor of 2 dimensions [batch size x sequence length].  Not yet implemented.\n\n        :return: The input after applying the reduction operation to sequence dimension.\n        '
        return self._reduce_obj(inputs, mask=mask)

    @property
    def input_shape(self) -> torch.Size:
        if False:
            for i in range(10):
                print('nop')
        'Returns size of the input tensor without the batch dimension.'
        if self._encoding_size is None:
            return torch.Size([self._max_sequence_length])
        else:
            return torch.Size([self._max_sequence_length, self._encoding_size])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            while True:
                i = 10
        'Returns size of the output tensor without the batch dimension.'
        input_shape = self.input_shape
        if self._reduce_mode in {None, 'none', 'None'}:
            return input_shape
        elif self._reduce_mode == 'concat':
            if len(input_shape) > 1:
                return input_shape[:-2] + (input_shape[-1] * input_shape[-2],)
            return input_shape
        else:
            return input_shape[1:]

class ReduceLast(torch.nn.Module):

    def forward(self, inputs, mask=None):
        if False:
            while True:
                i = 10
        batch_size = inputs.shape[0]
        sequence_length = sequence_length_3D(inputs) - 1
        sequence_length[sequence_length < 0] = 0
        gathered = inputs[torch.arange(batch_size), sequence_length.type(torch.int64)]
        return gathered

class ReduceSum(torch.nn.Module):

    def forward(self, inputs, mask=None):
        if False:
            return 10
        return torch.sum(inputs, dim=1)

class ReduceMean(torch.nn.Module):

    def forward(self, inputs, mask=None):
        if False:
            print('Hello World!')
        return torch.mean(inputs, dim=1)

class ReduceMax(torch.nn.Module):

    def forward(self, inputs, mask=None):
        if False:
            for i in range(10):
                print('nop')
        return torch.amax(inputs, dim=1)

class ReduceConcat(torch.nn.Module):

    def forward(self, inputs, mask=None):
        if False:
            while True:
                i = 10
        if inputs.dim() > 2:
            return inputs.reshape(-1, inputs.shape[-1] * inputs.shape[-2])
        return inputs

class ReduceNone(torch.nn.Module):

    def forward(self, inputs, mask=None):
        if False:
            print('Hello World!')
        return inputs
reduce_mode_registry = {'last': ReduceLast, 'sum': ReduceSum, 'mean': ReduceMean, 'avg': ReduceMean, 'max': ReduceMax, 'concat': ReduceConcat, 'attention': FeedForwardAttentionReducer, 'none': ReduceNone, 'None': ReduceNone, None: ReduceNone}