""" Fast Tokenization class for model DeBERTa."""
import json
from typing import List, Optional, Tuple
from tokenizers import pre_tokenizers
from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_deberta import DebertaTokenizer
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'merges_file': 'merges.txt', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'microsoft/deberta-base': 'https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json', 'microsoft/deberta-large': 'https://huggingface.co/microsoft/deberta-large/resolve/main/vocab.json', 'microsoft/deberta-xlarge': 'https://huggingface.co/microsoft/deberta-xlarge/resolve/main/vocab.json', 'microsoft/deberta-base-mnli': 'https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/vocab.json', 'microsoft/deberta-large-mnli': 'https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/vocab.json', 'microsoft/deberta-xlarge-mnli': 'https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/vocab.json'}, 'merges_file': {'microsoft/deberta-base': 'https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt', 'microsoft/deberta-large': 'https://huggingface.co/microsoft/deberta-large/resolve/main/merges.txt', 'microsoft/deberta-xlarge': 'https://huggingface.co/microsoft/deberta-xlarge/resolve/main/merges.txt', 'microsoft/deberta-base-mnli': 'https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/merges.txt', 'microsoft/deberta-large-mnli': 'https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/merges.txt', 'microsoft/deberta-xlarge-mnli': 'https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/merges.txt'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'microsoft/deberta-base': 512, 'microsoft/deberta-large': 512, 'microsoft/deberta-xlarge': 512, 'microsoft/deberta-base-mnli': 512, 'microsoft/deberta-large-mnli': 512, 'microsoft/deberta-xlarge-mnli': 512}
PRETRAINED_INIT_CONFIGURATION = {'microsoft/deberta-base': {'do_lower_case': False}, 'microsoft/deberta-large': {'do_lower_case': False}}

class DebertaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" DeBERTa tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import DebertaTokenizerFast

    >>> tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
    >>> tokenizer("Hello world")["input_ids"]
    [1, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [1, 20920, 232, 2]
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
            The path to a tokenizer file to use instead of the vocab file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Deberta tokenizer detect beginning of words by the preceding space).
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ['input_ids', 'attention_mask', 'token_type_ids']
    slow_tokenizer_class = DebertaTokenizer

    def __init__(self, vocab_file=None, merges_file=None, tokenizer_file=None, errors='replace', bos_token='[CLS]', eos_token='[SEP]', sep_token='[SEP]', cls_token='[CLS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]', add_prefix_space=False, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(vocab_file, merges_file, tokenizer_file=tokenizer_file, errors=errors, bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, sep_token=sep_token, cls_token=cls_token, pad_token=pad_token, mask_token=mask_token, add_prefix_space=add_prefix_space, **kwargs)
        self.add_bos_token = kwargs.pop('add_bos_token', False)
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get('add_prefix_space', add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop('type'))
            pre_tok_state['add_prefix_space'] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
        self.add_prefix_space = add_prefix_space

    @property
    def mask_token(self) -> str:
        if False:
            return 10
        '\n        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not\n        having been set.\n\n        Deberta tokenizer has a special mask token to be used in the fill-mask pipeline. The mask token will greedily\n        comprise the space before the *[MASK]*.\n        '
        if self._mask_token is None:
            if self.verbose:
                logger.error('Using mask_token, but it is not set yet.')
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        if False:
            return 10
        '\n        Overriding the default behavior of the mask token to have it eat the space before it.\n        '
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. A DeBERTa sequence has the following format:\n\n        - single sequence: [CLS] X [SEP]\n        - pair of sequences: [CLS] A [SEP] B [SEP]\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa\n        sequence pair mask has the following format:\n\n        ```\n        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1\n        | first sequence    | second sequence |\n        ```\n\n        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)