from typing import Any, Dict, List
import torch
from allennlp.common import Registrable
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
IndexedTokenList = Dict[str, List[Any]]

class TokenIndexer(Registrable):
    """
    A `TokenIndexer` determines how string tokens get represented as arrays of indices in a model.
    This class both converts strings into numerical values, with the help of a
    :class:`~allennlp.data.vocabulary.Vocabulary`, and it produces actual arrays.

    Tokens can be represented as single IDs (e.g., the word "cat" gets represented by the number
    34), or as lists of character IDs (e.g., "cat" gets represented by the numbers [23, 10, 18]),
    or in some other way that you can come up with (e.g., if you have some structured input you
    want to represent in a special way in your data arrays, you can do that here).

    # Parameters

    token_min_padding_length : `int`, optional (default=`0`)
        The minimum padding length required for the :class:`TokenIndexer`. For example,
        the minimum padding length of :class:`SingleIdTokenIndexer` is the largest size of
        filter when using :class:`CnnEncoder`.
        Note that if you set this for one TokenIndexer, you likely have to set it for all
        :class:`TokenIndexer` for the same field, otherwise you'll get mismatched tensor sizes.
    """
    default_implementation = 'single_id'
    has_warned_for_as_padded_tensor = False

    def __init__(self, token_min_padding_length: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._token_min_padding_length: int = token_min_padding_length

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if False:
            print('Hello World!')
        '\n        The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training\n        data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,\n        token).  This method takes a token and a dictionary of counts and increments counts for\n        whatever vocabulary items are present in the token.  If this is a single token ID\n        representation, the vocabulary item is likely the token itself.  If this is a token\n        characters representation, the vocabulary items are all of the characters in the token.\n        '
        raise NotImplementedError

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        if False:
            while True:
                i = 10
        '\n        Takes a list of tokens and converts them to an `IndexedTokenList`.\n        This could be just an ID for each token from the vocabulary.\n        Or it could split each token into characters and return one ID per character.\n        Or (for instance, in the case of byte-pair encoding) there might not be a clean\n        mapping from individual tokens to indices, and the `IndexedTokenList` could be a complex\n        data structure.\n        '
        raise NotImplementedError

    def indices_to_tokens(self, indexed_tokens: IndexedTokenList, vocabulary: Vocabulary) -> List[Token]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Inverse operations of tokens_to_indices. Takes an `IndexedTokenList` and converts it back\n        into a list of tokens.\n        '
        raise NotImplementedError

    def get_empty_token_list(self) -> IndexedTokenList:
        if False:
            i = 10
            return i + 15
        '\n        Returns an `already indexed` version of an empty token list.  This is typically just an\n        empty list for whatever keys are used in the indexer.\n        '
        raise NotImplementedError

    def get_padding_lengths(self, indexed_tokens: IndexedTokenList) -> Dict[str, int]:
        if False:
            return 10
        "\n        This method returns a padding dictionary for the given `indexed_tokens` specifying all\n        lengths that need padding.  If all you have is a list of single ID tokens, this is just the\n        length of the list, and that's what the default implementation will give you.  If you have\n        something more complicated, like a list of character ids for token, you'll need to override\n        this.\n        "
        padding_lengths = {}
        for (key, token_list) in indexed_tokens.items():
            padding_lengths[key] = max(len(token_list), self._token_min_padding_length)
        return padding_lengths

    def as_padded_tensor_dict(self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        if False:
            while True:
                i = 10
        '\n        This method pads a list of tokens given the input padding lengths (which could actually\n        truncate things, depending on settings) and returns that padded list of input tokens as a\n        `Dict[str, torch.Tensor]`.  This is a dictionary because there should be one key per\n        argument that the `TokenEmbedder` corresponding to this class expects in its `forward()`\n        method (where the argument name in the `TokenEmbedder` needs to make the key in this\n        dictionary).\n\n        The base class implements the case when all you want to do is create a padded `LongTensor`\n        for every list in the `tokens` dictionary.  If your `TokenIndexer` needs more complex\n        logic than that, you need to override this method.\n        '
        tensor_dict = {}
        for (key, val) in tokens.items():
            if val and isinstance(val[0], bool):
                tensor = torch.BoolTensor(pad_sequence_to_length(val, padding_lengths[key], default_value=lambda : False))
            else:
                tensor = torch.LongTensor(pad_sequence_to_length(val, padding_lengths[key]))
            tensor_dict[key] = tensor
        return tensor_dict

    def __eq__(self, other) -> bool:
        if False:
            while True:
                i = 10
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented