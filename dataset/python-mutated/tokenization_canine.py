"""Tokenization classes for CANINE."""
from typing import Dict, List, Optional
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
logger = logging.get_logger(__name__)
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'nielsr/canine-s': 2048}
UNICODE_VOCAB_SIZE = 1114112
PAD = 0
CLS = 57344
SEP = 57345
BOS = 57346
MASK = 57347
RESERVED = 57348
SPECIAL_CODEPOINTS: Dict[int, str] = {CLS: '[CLS]', SEP: '[SEP]', BOS: '[BOS]', MASK: '[MASK]', PAD: '[PAD]', RESERVED: '[RESERVED]'}
SPECIAL_CODEPOINTS_BY_NAME: Dict[str, int] = {name: codepoint for (codepoint, name) in SPECIAL_CODEPOINTS.items()}

class CanineTokenizer(PreTrainedTokenizer):
    """
    Construct a CANINE tokenizer (i.e. a character splitter). It turns text into a sequence of characters, and then
    converts each character into its Unicode code point.

    [`CanineTokenizer`] inherits from [`PreTrainedTokenizer`].

    Refer to superclass [`PreTrainedTokenizer`] for usage examples and documentation concerning parameters.

    Args:
        model_max_length (`int`, *optional*, defaults to 2048):
                The maximum sentence length the model accepts.
    """
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, bos_token=chr(CLS), eos_token=chr(SEP), sep_token=chr(SEP), cls_token=chr(CLS), pad_token=chr(PAD), mask_token=chr(MASK), add_prefix_space=False, model_max_length=2048, **kwargs):
        if False:
            i = 10
            return i + 15
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        self._special_codepoints: Dict[str, int] = {}
        for (codepoint, name) in SPECIAL_CODEPOINTS.items():
            self._special_codepoints[name] = codepoint
        self._special_codepoint_strings: Dict[int, str] = {codepoint: name for (name, codepoint) in self._special_codepoints.items()}
        self._unicode_vocab_size = UNICODE_VOCAB_SIZE
        self._num_special_tokens = len(self._special_codepoints)
        super().__init__(bos_token=bos_token, eos_token=eos_token, sep_token=sep_token, cls_token=cls_token, pad_token=pad_token, mask_token=mask_token, add_prefix_space=add_prefix_space, model_max_length=model_max_length, **kwargs)

    @property
    def vocab_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._unicode_vocab_size

    def get_vocab(self):
        if False:
            print('Hello World!')
        vocab = {chr(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        if False:
            return 10
        'Tokenize a string (i.e. perform character splitting).'
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        if False:
            while True:
                i = 10
        'Converts a token (i.e. a Unicode character) in an id (i.e. its integer Unicode code point value).'
        try:
            return ord(token)
        except TypeError:
            raise ValueError(f"invalid token: '{token}'")

    def _convert_id_to_token(self, index: int) -> str:
        if False:
            while True:
                i = 10
        "\n        Converts a Unicode code point (integer) in a token (str). In case it's a special code point, convert to\n        human-readable format.\n        "
        try:
            if index in SPECIAL_CODEPOINTS:
                return SPECIAL_CODEPOINTS[index]
            return chr(index)
        except TypeError:
            raise ValueError(f'invalid id: {index}')

    def convert_tokens_to_string(self, tokens):
        if False:
            return 10
        return ''.join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            while True:
                i = 10
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. A CANINE sequence has the following format:\n\n        - single sequence: `[CLS] X [SEP]`\n        - pair of sequences: `[CLS] A [SEP] B [SEP]`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            i = 10
            return i + 15
        '\n        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding\n        special tokens using the tokenizer `prepare_for_model` method.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n            already_has_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not the token list is already formatted with special tokens for the model.\n\n        Returns:\n            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.\n        '
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
        result = [1] + [0] * len(token_ids_0) + [1]
        if token_ids_1 is not None:
            result += [0] * len(token_ids_1) + [1]
        return result

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A CANINE\n        sequence pair mask has the following format:\n\n        ```\n        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1\n        | first sequence    | second sequence |\n        ```\n\n        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None):
        if False:
            while True:
                i = 10
        return ()