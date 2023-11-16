"""Tokenization classes for ESM."""
import os
from typing import List, Optional, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'facebook/esm2_t6_8M_UR50D': 'https://huggingface.co/facebook/esm2_t6_8M_UR50D/resolve/main/vocab.txt', 'facebook/esm2_t12_35M_UR50D': 'https://huggingface.co/facebook/esm2_t12_35M_UR50D/resolve/main/vocab.txt'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'facebook/esm2_t6_8M_UR50D': 1024, 'facebook/esm2_t12_35M_UR50D': 1024}

def load_vocab_file(vocab_file):
    if False:
        print('Hello World!')
    with open(vocab_file, 'r') as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]

class EsmTokenizer(PreTrainedTokenizer):
    """
    Constructs an ESM tokenizer.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, vocab_file, unk_token='<unk>', cls_token='<cls>', pad_token='<pad>', mask_token='<mask>', eos_token='<eos>', **kwargs):
        if False:
            print('Hello World!')
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {tok: ind for (ind, tok) in enumerate(self.all_tokens)}
        super().__init__(unk_token=unk_token, cls_token=cls_token, pad_token=pad_token, mask_token=mask_token, eos_token=eos_token, **kwargs)
        self.unique_no_split_tokens = self.all_tokens
        self._update_trie(self.unique_no_split_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        if False:
            while True:
                i = 10
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        if False:
            while True:
                i = 10
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _tokenize(self, text, **kwargs):
        if False:
            while True:
                i = 10
        return text.split()

    def get_vocab_size(self, with_added_tokens=False):
        if False:
            for i in range(10):
                print('nop')
        return len(self._id_to_token)

    def get_vocab(self):
        if False:
            return 10
        return {token: i for (i, token) in enumerate(self.all_tokens)}

    def token_to_id(self, token: str) -> int:
        if False:
            print('Hello World!')
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            i = 10
            return i + 15
        cls = [self.cls_token_id]
        sep = [self.eos_token_id]
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            else:
                return cls + token_ids_0 + sep
        elif self.eos_token_id is None:
            raise ValueError('Cannot tokenize multiple sequences when EOS token is not set!')
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0: List, token_ids_1: Optional[List]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding\n        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of ids of the first sequence.\n            token_ids_1 (`List[int]`, *optional*):\n                List of ids of the second sequence.\n            already_has_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not the token list is already formatted with special tokens for the model.\n\n        Returns:\n            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.\n        '
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError('You should not supply a second sequence if the provided sequence of ids is already formatted with special tokens for the model.')
            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = [1] + [0] * len(token_ids_0) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask

    def save_vocabulary(self, save_directory, filename_prefix):
        if False:
            print('Hello World!')
        vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + 'vocab.txt')
        with open(vocab_file, 'w') as f:
            f.write('\n'.join(self.all_tokens))
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        if False:
            print('Hello World!')
        return self.get_vocab_size(with_added_tokens=False)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool=False) -> int:
        if False:
            i = 10
            return i + 15
        return super()._add_tokens(new_tokens, special_tokens=True)