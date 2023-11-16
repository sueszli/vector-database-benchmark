"""Tokenization classes for OFA."""
from typing import List, Optional, Tuple
import json
from tokenizers import normalizers
from transformers import PreTrainedTokenizerFast
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
from transformers.utils import logging
from modelscope.utils.constant import ModelFile
from .tokenization_ofa import OFATokenizer, OFATokenizerZH
logger = logging.get_logger()
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'merges_file': 'merges.txt', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'ofa-base': 'https://huggingface.co/ofa-base/resolve/main/vocab.json'}, 'merges_file': {'ofa-base': 'https://huggingface.co/ofa-base/resolve/main/merges.txt'}, 'tokenizer_file': {'ofa-base': 'https://huggingface.co/ofa-base/resolve/main/tokenizer.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'ofa-base': 1024}
VOCAB_FILES_NAMES_ZH = {'vocab_file': ModelFile.VOCAB_FILE}
PRETRAINED_VOCAB_FILES_MAP_ZH = {'vocab_file': {'bert-base-chinese': 'https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES_ZH = {'ofa-base': 1024}
PRETRAINED_INIT_CONFIGURATION_ZH = {'bert-base-chinese': {'do_lower_case': True}}

class OFATokenizerFast(BartTokenizerFast):
    """
    Construct a "fast" OFA tokenizer (backed by HuggingFace's *tokenizers* library).

    [`~OFATokenizerFast`] is identical to [`BartTokenizerFast`] and runs end-to-end tokenization: punctuation splitting
    and wordpiece.

    Refer to superclass [`BartTokenizerFast`] for usage examples and documentation concerning parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = OFATokenizer

class OFATokenizerZHFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" OFA tokenizer (backed by HuggingFace's *tokenizers* library).

    [`~OFATokenizerFast`] is identical to [`BartTokenizerFast`] and runs end-to-end tokenization: punctuation splitting
    and wordpiece.

    Refer to superclass [`BartTokenizerFast`] for usage examples and documentation concerning parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES_ZH
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP_ZH
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION_ZH
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES_ZH
    slow_tokenizer_class = OFATokenizerZH

    def __init__(self, vocab_file=None, tokenizer_file=None, do_lower_case=True, bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>', tokenize_chinese_chars=True, strip_accents=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(vocab_file, tokenizer_file=tokenizer_file, do_lower_case=do_lower_case, bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, sep_token=sep_token, cls_token=cls_token, pad_token=pad_token, mask_token=mask_token, tokenize_chinese_chars=tokenize_chinese_chars, strip_accents=strip_accents, **kwargs)
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if normalizer_state.get('lowercase', do_lower_case) != do_lower_case or normalizer_state.get('strip_accents', strip_accents) != strip_accents or normalizer_state.get('handle_chinese_chars', tokenize_chinese_chars) != tokenize_chinese_chars:
            normalizer_class = getattr(normalizers, normalizer_state.pop('type'))
            normalizer_state['lowercase'] = do_lower_case
            normalizer_state['strip_accents'] = strip_accents
            normalizer_state['handle_chinese_chars'] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)
        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. A BERT sequence has the following format:\n\n        - single sequence: `[CLS] X [SEP]`\n        - pair of sequences: `[CLS] A [SEP] B [SEP]`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        if token_ids_1:
            output += token_ids_1 + [self.sep_token_id]
        return output

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence\n        pair mask has the following format:\n\n        ```\n        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1\n        | first sequence    | second sequence |\n        ```\n\n        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            return 10
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)