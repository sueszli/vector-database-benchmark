""" Tokenization class for model ByT5."""
import warnings
from typing import List, Optional, Tuple
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
logger = logging.get_logger(__name__)

class ByT5Tokenizer(PreTrainedTokenizer):
    """
    Construct a ByT5 tokenizer. ByT5 simply uses raw bytes utf-8 encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        extra_ids (`int`, *optional*, defaults to 125):
            Add a number of extra ids added to the end of the vocabulary for use as sentinels. These tokens are
            accessible as "<extra_id_{%d}>" where "{%d}" is a number between 0 and extra_ids-1. Extra tokens are
            indexed from the end of the vocabulary up to beginning ("<extra_id_0>" is the last token in the vocabulary
            like in ByT5 preprocessing see
            [here](https://github.com/google-research/text-to-text-transfer-transformer/blob/9fd7b14a769417be33bc6c850f9598764913c833/t5/data/preprocessors.py#L2117)).
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
    """
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, eos_token='</s>', unk_token='<unk>', pad_token='<pad>', extra_ids=125, additional_special_tokens=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [f'<extra_id_{i}>' for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None and (len(additional_special_tokens) > 0):
            extra_tokens = len(set(filter(lambda x: bool('extra_id' in str(x)), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(f'Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to ByT5Tokenizer. In this case the additional_special_tokens must include the extra_ids tokens')
        pad_token = AddedToken(pad_token, lstrip=True, rstrip=True) if isinstance(pad_token, str) else pad_token
        eos_token = AddedToken(eos_token, lstrip=True, rstrip=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=True, rstrip=True) if isinstance(unk_token, str) else unk_token
        self._added_tokens_decoder = {0: pad_token, 1: eos_token, 2: unk_token}
        self.offset = len(self._added_tokens_decoder)
        self._utf_vocab_size = 2 ** 8
        super().__init__(eos_token=eos_token, unk_token=unk_token, pad_token=pad_token, extra_ids=0, additional_special_tokens=additional_special_tokens, **kwargs)

    @property
    def vocab_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self._utf_vocab_size

    def get_vocab(self):
        if False:
            i = 10
            return i + 15
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding\n        special tokens using the tokenizer `prepare_for_model` method.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n            already_has_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not the token list is already formatted with special tokens for the model.\n\n        Returns:\n            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.\n        '
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
        if token_ids_1 is None:
            return [0] * len(token_ids_0) + [1]
        return [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        if False:
            return 10
        'Do not add eos again if user already added it.'
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(f'This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added.')
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            return 10
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. ByT5 does not\n        make use of token type ids, therefore a list of zeros is returned.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of zeros.\n        '
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            i = 10
            return i + 15
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. A sequence has the following format:\n\n        - single sequence: `X </s>`\n        - pair of sequences: `A </s> B </s>`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def _tokenize(self, text: str) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Take as input a string and return a list of strings (tokens) for words/sub-words'
        tokens = [chr(i) for i in text.encode('utf-8')]
        return tokens

    def _convert_token_to_id(self, token):
        if False:
            i = 10
            return i + 15
        'Converts a token (str) in an id using the vocab.'
        if len(token) != 1:
            token_id = None
        else:
            token_id = ord(token) + self.offset
        return token_id

    def _convert_id_to_token(self, index):
        if False:
            while True:
                i = 10
        'Converts an index (integer) in a token (str) using the vocab.'
        token = chr(index - self.offset)
        return token

    def convert_tokens_to_string(self, tokens):
        if False:
            print('Hello World!')
        'Converts a sequence of tokens (string) in a single string.'
        bstring = b''
        for token in tokens:
            if token in self.added_tokens_decoder:
                tok_string = self.added_tokens_decoder[token].encode('utf-8')
            elif token in self.added_tokens_encoder:
                tok_string = token.encode('utf-8')
            else:
                tok_string = bytes([ord(token)])
            bstring += tok_string
        string = bstring.decode('utf-8', errors='ignore')
        return string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            i = 10
            return i + 15
        return ()