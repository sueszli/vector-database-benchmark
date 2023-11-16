"""Tokenization classes for OpenAI GPT."""
import json
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import numpy as np
from ...utils import is_tf_available, is_torch_available, logging
if TYPE_CHECKING:
    if is_torch_available():
        import torch
    if is_tf_available():
        import tensorflow as tf
from tokenizers import pre_tokenizers
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from .tokenization_codegen import CodeGenTokenizer
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'merges_file': 'merges.txt', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'Salesforce/codegen-350M-mono': 'https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/vocab.json'}, 'merges_file': {'Salesforce/codegen-350M-mono': 'https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/merges.txt'}, 'tokenizer_file': {'Salesforce/codegen-350M-mono': 'https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/tokenizer.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'Salesforce/codegen-350M-mono': 2048}

class CodeGenTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" CodeGen tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import CodeGenTokenizerFast

    >>> tokenizer = CodeGenTokenizerFast.from_pretrained("Salesforce/codegen-350M-mono")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (CodeGen tokenizer detect beginning of words by the preceding space).
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ['input_ids', 'attention_mask']
    slow_tokenizer_class = CodeGenTokenizer

    def __init__(self, vocab_file=None, merges_file=None, tokenizer_file=None, unk_token='<|endoftext|>', bos_token='<|endoftext|>', eos_token='<|endoftext|>', add_prefix_space=False, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(vocab_file, merges_file, tokenizer_file=tokenizer_file, unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, add_prefix_space=add_prefix_space, **kwargs)
        if kwargs.pop('add_bos_token', False):
            model_id = kwargs.pop('name_or_path', '')
            raise ValueError(f"Currenty GPT2's fast tokenizer does NOT support adding a BOS token. Instead you should use GPT2's slow tokenizer class `CodeGenTokenizer` as follows: \n`CodeGenTokenizer.from_pretrained('{model_id}')`\nor\n`AutoTokenizer.from_pretrained('{model_id}', use_fast=False)`\nThis issue will be fixed soon, see: https://github.com/huggingface/tokenizers/pull/1005. so that the fast tokenizer works correctly.")
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get('add_prefix_space', add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop('type'))
            pre_tok_state['add_prefix_space'] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
        self.add_prefix_space = add_prefix_space

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        if False:
            while True:
                i = 10
        is_split_into_words = kwargs.get('is_split_into_words', False)
        assert self.add_prefix_space or not is_split_into_words, f'You need to instantiate {self.__class__.__name__} with add_prefix_space=True to use it with pretokenized inputs.'
        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        if False:
            while True:
                i = 10
        is_split_into_words = kwargs.get('is_split_into_words', False)
        assert self.add_prefix_space or not is_split_into_words, f'You need to instantiate {self.__class__.__name__} with add_prefix_space=True to use it with pretokenized inputs.'
        return super()._encode_plus(*args, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            while True:
                i = 10
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def decode(self, token_ids: Union[int, List[int], 'np.ndarray', 'torch.Tensor', 'tf.Tensor'], skip_special_tokens: bool=False, clean_up_tokenization_spaces: bool=None, truncate_before_pattern: Optional[List[str]]=None, **kwargs) -> str:
        if False:
            return 10
        '\n        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special\n        tokens and clean up tokenization spaces.\n\n        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.\n\n        Args:\n            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):\n                List of tokenized input ids. Can be obtained using the `__call__` method.\n            skip_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not to remove special tokens in the decoding.\n            clean_up_tokenization_spaces (`bool`, *optional*):\n                Whether or not to clean up the tokenization spaces. If `None`, will default to\n                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).\n            truncate_before_pattern (`List[str]`, *optional*, defaults to `None`):\n                A list of regular expression strings that will be used to truncate the returned string. This can be\n                used to remove extra pieces of code (e.g. truncate if observing a comment symbol "#" at the beginning\n                of a new line). An example pattern could be `["^#", re.escape("<|endoftext|>"), "^\'\'\'", "\n\n\n"]`.\n            kwargs (additional keyword arguments, *optional*):\n                Will be passed to the underlying model specific decode method.\n\n        Returns:\n            `str`: The decoded sentence.\n        '
        decoded_text = super().decode(token_ids=token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)
        if truncate_before_pattern is not None and len(truncate_before_pattern) > 0:
            decoded_text = self.truncate(decoded_text, truncate_before_pattern)
        return decoded_text

    def truncate(self, completion, truncate_before_pattern):
        if False:
            i = 10
            return i + 15

        def find_re(string, pattern, start_pos):
            if False:
                while True:
                    i = 10
            m = pattern.search(string, start_pos)
            return m.start() if m else -1
        terminals = [re.compile(pattern, re.MULTILINE) for pattern in truncate_before_pattern]
        prints = list(re.finditer('^print', completion, re.MULTILINE))
        if len(prints) > 1:
            completion = completion[:prints[1].start()]
        defs = list(re.finditer('^def', completion, re.MULTILINE))
        if len(defs) > 1:
            completion = completion[:defs[1].start()]
        start_pos = 0
        terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
        if len(terminals_pos) > 0:
            return completion[:min(terminals_pos)]
        else:
            return completion