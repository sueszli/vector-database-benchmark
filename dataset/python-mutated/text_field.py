"""
A `TextField` represents a string of text, the kind that you might want to represent with
standard word vectors, or pass through an LSTM.
"""
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Iterator
import textwrap
from spacy.tokens import Token as SpacyToken
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util
TextFieldTensors = Dict[str, Dict[str, torch.Tensor]]

class TextField(SequenceField[TextFieldTensors]):
    """
    This `Field` represents a list of string tokens.  Before constructing this object, you need
    to tokenize raw strings using a :class:`~allennlp.data.tokenizers.tokenizer.Tokenizer`.

    Because string tokens can be represented as indexed arrays in a number of ways, we also take a
    dictionary of :class:`~allennlp.data.token_indexers.token_indexer.TokenIndexer`
    objects that will be used to convert the tokens into indices.
    Each `TokenIndexer` could represent each token as a single ID, or a list of character IDs, or
    something else.

    This field will get converted into a dictionary of arrays, one for each `TokenIndexer`.  A
    `SingleIdTokenIndexer` produces an array of shape (num_tokens,), while a
    `TokenCharactersIndexer` produces an array of shape (num_tokens, num_characters).
    """
    __slots__ = ['tokens', '_token_indexers', '_indexed_tokens']

    def __init__(self, tokens: List[Token], token_indexers: Optional[Dict[str, TokenIndexer]]=None) -> None:
        if False:
            return 10
        self.tokens = tokens
        self._token_indexers = token_indexers
        self._indexed_tokens: Optional[Dict[str, IndexedTokenList]] = None
        if not all((isinstance(x, (Token, SpacyToken)) for x in tokens)):
            raise ConfigurationError('TextFields must be passed Tokens. Found: {} with types {}.'.format(tokens, [type(x) for x in tokens]))

    @property
    def token_indexers(self) -> Dict[str, TokenIndexer]:
        if False:
            i = 10
            return i + 15
        if self._token_indexers is None:
            raise ValueError("TextField's token_indexers have not been set.\nDid you forget to call DatasetReader.apply_token_indexers(instance) on your instance?\nIf apply_token_indexers() is being called but you're still seeing this error, it may not be implemented correctly.")
        return self._token_indexers

    @token_indexers.setter
    def token_indexers(self, token_indexers: Dict[str, TokenIndexer]) -> None:
        if False:
            return 10
        self._token_indexers = token_indexers

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if False:
            print('Hello World!')
        for indexer in self.token_indexers.values():
            for token in self.tokens:
                indexer.count_vocab_items(token, counter)

    def index(self, vocab: Vocabulary):
        if False:
            return 10
        self._indexed_tokens = {}
        for (indexer_name, indexer) in self.token_indexers.items():
            self._indexed_tokens[indexer_name] = indexer.tokens_to_indices(self.tokens, vocab)

    def get_padding_lengths(self) -> Dict[str, int]:
        if False:
            return 10
        '\n        The `TextField` has a list of `Tokens`, and each `Token` gets converted into arrays by\n        (potentially) several `TokenIndexers`.  This method gets the max length (over tokens)\n        associated with each of these arrays.\n        '
        if self._indexed_tokens is None:
            raise ConfigurationError('You must call .index(vocabulary) on a field before determining padding lengths.')
        padding_lengths = {}
        for (indexer_name, indexer) in self.token_indexers.items():
            indexer_lengths = indexer.get_padding_lengths(self._indexed_tokens[indexer_name])
            for (key, length) in indexer_lengths.items():
                padding_lengths[f'{indexer_name}___{key}'] = length
        return padding_lengths

    def sequence_length(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.tokens)

    def as_tensor(self, padding_lengths: Dict[str, int]) -> TextFieldTensors:
        if False:
            i = 10
            return i + 15
        if self._indexed_tokens is None:
            raise ConfigurationError('You must call .index(vocabulary) on a field before calling .as_tensor()')
        tensors = {}
        indexer_lengths: Dict[str, Dict[str, int]] = defaultdict(dict)
        for (key, value) in padding_lengths.items():
            (indexer_name, padding_key) = key.split('___')
            indexer_lengths[indexer_name][padding_key] = value
        for (indexer_name, indexer) in self.token_indexers.items():
            tensors[indexer_name] = indexer.as_padded_tensor_dict(self._indexed_tokens[indexer_name], indexer_lengths[indexer_name])
        return tensors

    def empty_field(self):
        if False:
            while True:
                i = 10
        text_field = TextField([], self._token_indexers)
        text_field._indexed_tokens = {}
        if self._token_indexers is not None:
            for (indexer_name, indexer) in self.token_indexers.items():
                text_field._indexed_tokens[indexer_name] = indexer.get_empty_token_list()
        return text_field

    def batch_tensors(self, tensor_list: List[TextFieldTensors]) -> TextFieldTensors:
        if False:
            for i in range(10):
                print('nop')
        indexer_lists: Dict[str, List[Dict[str, torch.Tensor]]] = defaultdict(list)
        for tensor_dict in tensor_list:
            for (indexer_name, indexer_output) in tensor_dict.items():
                indexer_lists[indexer_name].append(indexer_output)
        batched_tensors = {indexer_name: util.batch_tensor_dicts(indexer_outputs) for (indexer_name, indexer_outputs) in indexer_lists.items()}
        return batched_tensors

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        formatted_text = ''.join(('\t\t' + text + '\n' for text in textwrap.wrap(repr(self.tokens), 100)))
        if self._token_indexers is not None:
            indexers = {name: indexer.__class__.__name__ for (name, indexer) in self._token_indexers.items()}
            return f'TextField of length {self.sequence_length()} with text: \n {formatted_text} \t\tand TokenIndexers : {indexers}'
        else:
            return f'TextField of length {self.sequence_length()} with text: \n {formatted_text}'

    def __iter__(self) -> Iterator[Token]:
        if False:
            while True:
                i = 10
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> Token:
        if False:
            return 10
        return self.tokens[idx]

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self.tokens)

    def duplicate(self):
        if False:
            while True:
                i = 10
        "\n        Overrides the behavior of `duplicate` so that `self._token_indexers` won't\n        actually be deep-copied.\n\n        Not only would it be extremely inefficient to deep-copy the token indexers,\n        but it also fails in many cases since some tokenizers (like those used in\n        the 'transformers' lib) cannot actually be deep-copied.\n        "
        if self._token_indexers is not None:
            new = TextField(deepcopy(self.tokens), {k: v for (k, v) in self._token_indexers.items()})
        else:
            new = TextField(deepcopy(self.tokens))
        new._indexed_tokens = deepcopy(self._indexed_tokens)
        return new

    def human_readable_repr(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return [str(t) for t in self.tokens]