import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from ludwig.constants import TYPE
from ludwig.modules.initializer_modules import get_initializer
from ludwig.utils.data_utils import load_pretrained_embeddings
from ludwig.utils.torch_utils import get_torch_device, LudwigModule
logger = logging.getLogger(__name__)
DEVICE = get_torch_device()

def embedding_matrix(vocab: List[str], embedding_size: int, representation: str='dense', embeddings_trainable: bool=True, pretrained_embeddings: Optional[str]=None, force_embedding_size: bool=False, embedding_initializer: Optional[Union[str, Dict]]=None) -> Tuple[nn.Module, int]:
    if False:
        for i in range(10):
            print('nop')
    'Returns initialized torch.nn.Embedding module and embedding size.'
    vocab_size = len(vocab)
    if representation == 'dense':
        if pretrained_embeddings:
            embeddings_matrix = load_pretrained_embeddings(pretrained_embeddings, vocab)
            if embeddings_matrix.shape[-1] != embedding_size:
                if not force_embedding_size:
                    embedding_size = embeddings_matrix.shape[-1]
                    logger.info(f'Setting embedding size to be equal to {embeddings_matrix.shape[-1]}.')
                else:
                    raise ValueError(f'The size of the pretrained embeddings is {embeddings_matrix.shape[-1]}, but the specified embedding_size is {embedding_size}. Please change the embedding_size accordingly.')
            embedding_initializer_obj = torch.tensor(embeddings_matrix, dtype=torch.float32)
        else:
            if vocab_size < embedding_size and (not force_embedding_size):
                logger.info(f'  embedding_size ({embedding_size}) is greater than vocab_size ({vocab_size}). Setting embedding size to be equal to vocab_size.')
                embedding_size = vocab_size
            if embedding_initializer is not None:
                embedding_initializer_obj_ref = get_initializer(embedding_initializer)
            else:
                embedding_initializer_obj_ref = get_initializer({TYPE: 'uniform', 'a': -1.0, 'b': 1.0})
            embedding_initializer_obj = embedding_initializer_obj_ref([vocab_size, embedding_size])
        embeddings = embedding_initializer_obj
    elif representation == 'sparse':
        embedding_size = vocab_size
        embeddings = get_initializer('identity')([vocab_size, embedding_size])
        embeddings.requires_grad = False
    else:
        raise Exception(f'Embedding representation {representation} not supported.')
    embeddings = nn.Embedding.from_pretrained(embeddings, freeze=not embeddings_trainable)
    return (embeddings, embedding_size)

def embedding_matrix_on_device(vocab: List[str], embedding_size: int, representation: str='dense', embeddings_trainable: bool=True, pretrained_embeddings: Optional[str]=None, force_embedding_size: bool=False, embeddings_on_cpu: bool=False, embedding_initializer: Optional[str]=None) -> Tuple[nn.Module, int]:
    if False:
        print('Hello World!')
    (embeddings, embedding_size) = embedding_matrix(vocab, embedding_size, representation=representation, embeddings_trainable=embeddings_trainable, pretrained_embeddings=pretrained_embeddings, force_embedding_size=force_embedding_size, embedding_initializer=embedding_initializer)
    if embeddings_on_cpu:
        embeddings.to('cpu')
    elif not embeddings_on_cpu and torch.cuda.is_available():
        embeddings.to(device='cuda')
    return (embeddings, embedding_size)

class Embed(LudwigModule):
    """Module to embed Category, Date, and H3 data types."""

    def __init__(self, vocab: List[str], embedding_size: int, representation: str='dense', embeddings_trainable: bool=True, pretrained_embeddings: Optional[str]=None, force_embedding_size: bool=False, embeddings_on_cpu: bool=False, dropout: float=0.0, embedding_initializer: Optional[Union[str, Dict]]=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.supports_masking = True
        self.vocab_size = len(vocab)
        (self.embeddings, self.embedding_size) = embedding_matrix_on_device(vocab, embedding_size, representation=representation, embeddings_trainable=embeddings_trainable, pretrained_embeddings=pretrained_embeddings, force_embedding_size=force_embedding_size, embeddings_on_cpu=embeddings_on_cpu, embedding_initializer=embedding_initializer)
        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        if inputs.ndim != 2 or inputs.shape[1] != 1:
            raise RuntimeError(f'Embed only takes inputs of shape [batch x 1]. Received inputs with size: {inputs.size()}')
        embedded = self.embeddings(inputs.long())
        embedded = torch.squeeze(embedded, dim=1)
        if self.dropout:
            embedded = self.dropout(embedded)
        return embedded

    @property
    def input_shape(self) -> torch.Size:
        if False:
            i = 10
            return i + 15
        return torch.Size([1])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            i = 10
            return i + 15
        return torch.Size([self.embedding_size])

class EmbedSet(LudwigModule):
    """Module to embed Set data types, works on multi-hot encoded input."""

    def __init__(self, vocab: List[str], embedding_size: int, representation: str='dense', embeddings_trainable: bool=True, pretrained_embeddings: Optional[str]=None, force_embedding_size: bool=False, embeddings_on_cpu: bool=False, dropout: float=0.0, embedding_initializer: Optional[Union[str, Dict]]=None, aggregation_function: str='sum'):
        if False:
            return 10
        super().__init__()
        self.supports_masking = True
        self.vocab_size = len(vocab)
        (self.embeddings, self.embedding_size) = embedding_matrix_on_device(vocab, embedding_size, representation=representation, embeddings_trainable=embeddings_trainable, pretrained_embeddings=pretrained_embeddings, force_embedding_size=force_embedding_size, embeddings_on_cpu=embeddings_on_cpu, embedding_initializer=embedding_initializer)
        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if aggregation_function == 'sum':
            self.aggregation_function = torch.sum
        elif aggregation_function == 'avg':
            self.aggregation_function = torch.mean
        else:
            raise ValueError(f'Unsupported aggregation function {aggregation_function}')
        self.register_buffer('vocab_indices', torch.arange(self.vocab_size))

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Params:\n            inputs: Boolean multi-hot tensor of size [batch x vocab_size], where\n                    inputs[b, i] indicates that token i is present in sample b.\n        '
        inputs = inputs.int() * self.vocab_indices
        embedded = self.embeddings(inputs.long())
        mask = torch.unsqueeze(inputs, -1)
        embedded = embedded * mask
        embedded = self.aggregation_function(embedded, dim=1)
        if self.dropout:
            embedded = self.dropout(embedded)
        return embedded

    @property
    def input_shape(self) -> torch.Size:
        if False:
            i = 10
            return i + 15
        return torch.Size([self.vocab_size])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            for i in range(10):
                print('nop')
        return torch.Size([self.embedding_size])

    @property
    def input_dtype(self):
        if False:
            i = 10
            return i + 15
        return torch.bool

class EmbedWeighted(LudwigModule):
    """Module to embed Bag data type, works on input of token frequencies."""

    def __init__(self, vocab: List[str], embedding_size: int, representation: str='dense', embeddings_trainable: bool=True, pretrained_embeddings: Optional[str]=None, force_embedding_size: bool=False, embeddings_on_cpu: bool=False, dropout: float=0.0, embedding_initializer: Optional[str]=None):
        if False:
            print('Hello World!')
        super().__init__()
        (self.embeddings, self.embedding_size) = embedding_matrix_on_device(vocab, embedding_size, representation=representation, embeddings_trainable=embeddings_trainable, pretrained_embeddings=pretrained_embeddings, force_embedding_size=force_embedding_size, embeddings_on_cpu=embeddings_on_cpu, embedding_initializer=embedding_initializer)
        self.vocab_size = len(vocab)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.register_buffer('vocab_indices', torch.arange(self.vocab_size, dtype=torch.int32))

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Params:\n            inputs: Tensor of frequencies, where inputs[b, i] represents\n                    frequency of token i in sample b of batch.\n        '
        signed_input = (inputs != 0).type(torch.int32)
        multiple_hot_indexes = signed_input * self.vocab_indices
        embedded = self.embeddings(multiple_hot_indexes)
        mask = torch.unsqueeze(inputs, -1)
        weighted_embedded = embedded * mask
        embedded_reduced = torch.sum(weighted_embedded, dim=1)
        if self.dropout:
            embedded_reduced = self.dropout(embedded_reduced)
        return embedded_reduced

    @property
    def input_shape(self) -> torch.Size:
        if False:
            for i in range(10):
                print('nop')
        return torch.Size([self.vocab_size])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            print('Hello World!')
        return torch.Size([self.embedding_size])

class EmbedSequence(LudwigModule):

    def __init__(self, vocab: List[str], embedding_size: int, max_sequence_length: int, representation: str='dense', embeddings_trainable: bool=True, pretrained_embeddings: Optional[str]=None, force_embedding_size: bool=False, embeddings_on_cpu: bool=False, dropout: float=0.0, embedding_initializer: Optional[str]=None):
        if False:
            return 10
        super().__init__()
        self.supports_masking = True
        self.vocab_size = len(vocab)
        self.max_sequence_length = max_sequence_length
        (self.embeddings, self.embedding_size) = embedding_matrix_on_device(vocab, embedding_size, representation=representation, embeddings_trainable=embeddings_trainable, pretrained_embeddings=pretrained_embeddings, force_embedding_size=force_embedding_size, embeddings_on_cpu=embeddings_on_cpu, embedding_initializer=embedding_initializer)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor]=None):
        if False:
            while True:
                i = 10
        if inputs.dtype not in [torch.int, torch.long]:
            raise RuntimeError(f'Expected tensor of type torch.int or torch.long as input.Received {inputs.dtype} instead.')
        embedded = self.embeddings(inputs)
        if self.dropout:
            embedded = self.dropout(embedded)
        return embedded

    @property
    def input_shape(self) -> torch.Size:
        if False:
            while True:
                i = 10
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            i = 10
            return i + 15
        return torch.Size([self.max_sequence_length, self.embedding_size])

class TokenAndPositionEmbedding(LudwigModule):

    def __init__(self, max_sequence_length, vocab, embedding_size, representation='dense', embeddings_trainable=True, pretrained_embeddings=None, force_embedding_size=False, embeddings_on_cpu=False, dropout=0.0, embedding_initializer=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_size = embedding_size
        self.token_embed = EmbedSequence(vocab=vocab, embedding_size=embedding_size, max_sequence_length=max_sequence_length, representation=representation, embeddings_trainable=embeddings_trainable, pretrained_embeddings=pretrained_embeddings, force_embedding_size=force_embedding_size, embeddings_on_cpu=embeddings_on_cpu, dropout=dropout, embedding_initializer=embedding_initializer)
        self.position_embed = nn.Embedding(num_embeddings=max_sequence_length, embedding_dim=self.token_embed.embedding_size)
        self.register_buffer('positions', torch.arange(0, max_sequence_length))

    @property
    def input_shape(self) -> torch.Size:
        if False:
            while True:
                i = 10
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if False:
            while True:
                i = 10
        return self.token_embed.output_shape

    def forward(self, inputs, mask: Optional[torch.Tensor]=None):
        if False:
            for i in range(10):
                print('nop')
        positions_hidden = self.position_embed(self.positions)
        token_hidden = self.token_embed(inputs)
        return token_hidden + positions_hidden