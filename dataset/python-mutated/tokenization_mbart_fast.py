import os
from shutil import copyfile
from typing import List, Optional, Tuple
from tokenizers import processors
from ...tokenization_utils import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
if is_sentencepiece_available():
    from .tokenization_mbart import MBartTokenizer
else:
    MBartTokenizer = None
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'sentencepiece.bpe.model', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'facebook/mbart-large-en-ro': 'https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/sentencepiece.bpe.model', 'facebook/mbart-large-cc25': 'https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentencepiece.bpe.model'}, 'tokenizer_file': {'facebook/mbart-large-en-ro': 'https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/tokenizer.json', 'facebook/mbart-large-cc25': 'https://huggingface.co/facebook/mbart-large-cc25/resolve/main/tokenizer.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'facebook/mbart-large-en-ro': 1024, 'facebook/mbart-large-cc25': 1024}
FAIRSEQ_LANGUAGE_CODES = ['ar_AR', 'cs_CZ', 'de_DE', 'en_XX', 'es_XX', 'et_EE', 'fi_FI', 'fr_XX', 'gu_IN', 'hi_IN', 'it_IT', 'ja_XX', 'kk_KZ', 'ko_KR', 'lt_LT', 'lv_LV', 'my_MM', 'ne_NP', 'nl_XX', 'ro_RO', 'ru_RU', 'si_LK', 'tr_TR', 'vi_VN', 'zh_CN']

class MBartTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" MBART tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.

    Examples:

    ```python
    >>> from transformers import MBartTokenizerFast

    >>> tokenizer = MBartTokenizerFast.from_pretrained(
    ...     "facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO"
    ... )
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="pt")
    ```"""
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ['input_ids', 'attention_mask']
    slow_tokenizer_class = MBartTokenizer
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(self, vocab_file=None, tokenizer_file=None, bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>', src_lang=None, tgt_lang=None, additional_special_tokens=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        _additional_special_tokens = FAIRSEQ_LANGUAGE_CODES.copy()
        if additional_special_tokens is not None:
            _additional_special_tokens.extend([t for t in additional_special_tokens if t not in _additional_special_tokens])
        super().__init__(vocab_file=vocab_file, tokenizer_file=tokenizer_file, bos_token=bos_token, eos_token=eos_token, sep_token=sep_token, cls_token=cls_token, unk_token=unk_token, pad_token=pad_token, mask_token=mask_token, src_lang=src_lang, tgt_lang=tgt_lang, additional_special_tokens=_additional_special_tokens, **kwargs)
        self.vocab_file = vocab_file
        self.lang_code_to_id = {lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES}
        self._src_lang = src_lang if src_lang is not None else 'en_XX'
        self.cur_lang_code = self.convert_tokens_to_ids(self._src_lang)
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def can_save_slow_tokenizer(self) -> bool:
        if False:
            return 10
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    @property
    def src_lang(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        if False:
            print('Hello World!')
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. The special tokens depend on calling set_lang.\n\n        An MBART sequence has the following format, where `X` represents the sequence:\n\n        - `input_ids` (for encoder) `X [eos, src_lang_code]`\n        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`\n\n        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a\n        separator.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            while True:
                i = 10
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. mBART does not\n        make use of token type ids, therefore a list of zeros is returned.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of zeros.\n\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs):
        if False:
            while True:
                i = 10
        'Used by translation pipeline, to prepare inputs for the generate function'
        if src_lang is None or tgt_lang is None:
            raise ValueError('Translation requires a `src_lang` and a `tgt_lang` for this model')
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs['forced_bos_token_id'] = tgt_lang_id
        return inputs

    def prepare_seq2seq_batch(self, src_texts: List[str], src_lang: str='en_XX', tgt_texts: Optional[List[str]]=None, tgt_lang: str='ro_RO', **kwargs) -> BatchEncoding:
        if False:
            i = 10
            return i + 15
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        if False:
            return 10
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        if False:
            while True:
                i = 10
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code].'
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)
        self._tokenizer.post_processor = processors.TemplateProcessing(single=prefix_tokens_str + ['$A'] + suffix_tokens_str, pair=prefix_tokens_str + ['$A', '$B'] + suffix_tokens_str, special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)))

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        if False:
            while True:
                i = 10
        'Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code].'
        self.cur_lang_code = self.convert_tokens_to_ids(lang)
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)
        self._tokenizer.post_processor = processors.TemplateProcessing(single=prefix_tokens_str + ['$A'] + suffix_tokens_str, pair=prefix_tokens_str + ['$A', '$B'] + suffix_tokens_str, special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)))

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            while True:
                i = 10
        if not self.can_save_slow_tokenizer:
            raise ValueError('Your fast tokenizer does not have the necessary information to save the vocabulary for a slow tokenizer.')
        if not os.path.isdir(save_directory):
            logger.error(f'Vocabulary path ({save_directory}) should be a directory.')
            return
        out_vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        return (out_vocab_file,)