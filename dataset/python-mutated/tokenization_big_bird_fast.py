""" Tokenization classes for Big Bird model."""
import os
from shutil import copyfile
from typing import List, Optional, Tuple
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
if is_sentencepiece_available():
    from .tokenization_big_bird import BigBirdTokenizer
else:
    BigBirdTokenizer = None
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'spiece.model', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'google/bigbird-roberta-base': 'https://huggingface.co/google/bigbird-roberta-base/resolve/main/spiece.model', 'google/bigbird-roberta-large': 'https://huggingface.co/google/bigbird-roberta-large/resolve/main/spiece.model', 'google/bigbird-base-trivia-itc': 'https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/spiece.model'}, 'tokenizer_file': {'google/bigbird-roberta-base': 'https://huggingface.co/google/bigbird-roberta-base/resolve/main/tokenizer.json', 'google/bigbird-roberta-large': 'https://huggingface.co/google/bigbird-roberta-large/resolve/main/tokenizer.json', 'google/bigbird-base-trivia-itc': 'https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/tokenizer.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'google/bigbird-roberta-base': 4096, 'google/bigbird-roberta-large': 4096, 'google/bigbird-base-trivia-itc': 4096}
SPIECE_UNDERLINE = '▁'

class BigBirdTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" BigBird tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This
    tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
            that is used for the end of sequence. The token used is the `sep_token`.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = BigBirdTokenizer
    model_input_names = ['input_ids', 'attention_mask']
    prefix_tokens: List[int] = []

    def __init__(self, vocab_file=None, tokenizer_file=None, unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', sep_token='[SEP]', mask_token='[MASK]', cls_token='[CLS]', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        super().__init__(vocab_file, tokenizer_file=tokenizer_file, bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, **kwargs)
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        if False:
            print('Hello World!')
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. An BigBird sequence has the following format:\n\n        - single sequence: `[CLS] X [SEP]`\n        - pair of sequences: `[CLS] A [SEP] B [SEP]`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            return 10
        '\n        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding\n        special tokens using the tokenizer `prepare_for_model` method.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of ids.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n            already_has_special_tokens (`bool`, *optional*, defaults to `False`):\n                Set to True if the token list is already formatted with special tokens for the model\n\n        Returns:\n            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.\n        '
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError('You should not supply a second sequence if the provided sequence of ids is already formatted with special tokens for the model.')
            return [1 if x in [self.sep_token_id, self.cls_token_id] else 0 for x in token_ids_0]
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            i = 10
            return i + 15
        '\n        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT\n        sequence pair mask has the following format:\n\n        ```\n        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1\n        | first sequence    | second sequence |\n        ```\n\n        if token_ids_1 is None, only returns the first portion of the mask (0s).\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of ids.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            i = 10
            return i + 15
        if not self.can_save_slow_tokenizer:
            raise ValueError('Your fast tokenizer does not have the necessary information to save the vocabulary for a slow tokenizer.')
        if not os.path.isdir(save_directory):
            logger.error(f'Vocabulary path ({save_directory}) should be a directory')
            return
        out_vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        return (out_vocab_file,)