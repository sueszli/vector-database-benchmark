import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
from ...utils import logging
logger = logging.get_logger(__name__)
SPIECE_UNDERLINE = '▁'
VOCAB_FILES_NAMES = {'vocab_file': 'sentencepiece.bpe.model'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'facebook/mbart-large-50-one-to-many-mmt': 'https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt/resolve/main/sentencepiece.bpe.model'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'facebook/mbart-large-50-one-to-many-mmt': 1024}
FAIRSEQ_LANGUAGE_CODES = ['ar_AR', 'cs_CZ', 'de_DE', 'en_XX', 'es_XX', 'et_EE', 'fi_FI', 'fr_XX', 'gu_IN', 'hi_IN', 'it_IT', 'ja_XX', 'kk_KZ', 'ko_KR', 'lt_LT', 'lv_LV', 'my_MM', 'ne_NP', 'nl_XX', 'ro_RO', 'ru_RU', 'si_LK', 'tr_TR', 'vi_VN', 'zh_CN', 'af_ZA', 'az_AZ', 'bn_IN', 'fa_IR', 'he_IL', 'hr_HR', 'id_ID', 'ka_GE', 'km_KH', 'mk_MK', 'ml_IN', 'mn_MN', 'mr_IN', 'pl_PL', 'ps_AF', 'pt_XX', 'sv_SE', 'sw_KE', 'ta_IN', 'te_IN', 'th_TH', 'tl_XX', 'uk_UA', 'ur_PK', 'xh_ZA', 'gl_ES', 'sl_SI']

class MBart50Tokenizer(PreTrainedTokenizer):
    """
    Construct a MBart50 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        src_lang (`str`, *optional*):
            A string representing the source language.
        tgt_lang (`str`, *optional*):
            A string representing the target language.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

    Examples:

    ```python
    >>> from transformers import MBart50Tokenizer

    >>> tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
    >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
    >>> tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"
    >>> model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
    >>> # model(**model_inputs) should work
    ```"""
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ['input_ids', 'attention_mask']
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(self, vocab_file, src_lang=None, tgt_lang=None, eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>', sp_model_kwargs: Optional[Dict[str, Any]]=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        kwargs['additional_special_tokens'] = kwargs.get('additional_special_tokens', []) or []
        kwargs['additional_special_tokens'] += [code for code in FAIRSEQ_LANGUAGE_CODES if code not in kwargs['additional_special_tokens']]
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file
        self.fairseq_tokens_to_ids = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3}
        self.fairseq_offset = 1
        self.sp_model_size = len(self.sp_model)
        self.lang_code_to_id = {code: self.sp_model_size + i + self.fairseq_offset for (i, code) in enumerate(FAIRSEQ_LANGUAGE_CODES)}
        self.id_to_lang_code = {v: k for (k, v) in self.lang_code_to_id.items()}
        self.fairseq_tokens_to_ids['<mask>'] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset
        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {v: k for (k, v) in self.fairseq_tokens_to_ids.items()}
        super().__init__(src_lang=src_lang, tgt_lang=tgt_lang, eos_token=eos_token, unk_token=unk_token, sep_token=sep_token, cls_token=cls_token, pad_token=pad_token, mask_token=mask_token, sp_model_kwargs=self.sp_model_kwargs, **kwargs)
        self._src_lang = src_lang if src_lang is not None else 'en_XX'
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def vocab_size(self) -> int:
        if False:
            print('Hello World!')
        return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1

    @property
    def src_lang(self) -> str:
        if False:
            print('Hello World!')
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def __getstate__(self) -> Dict:
        if False:
            while True:
                i = 10
        state = self.__dict__.copy()
        state['sp_model'] = None
        return state

    def __setstate__(self, d: Dict) -> None:
        if False:
            while True:
                i = 10
        self.__dict__ = d
        if not hasattr(self, 'sp_model_kwargs'):
            self.sp_model_kwargs = {}
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def get_vocab(self) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        if False:
            return 10
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        if False:
            return 10
        'Converts a token (str) in an id using the vocab.'
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        if False:
            i = 10
            return i + 15
        'Converts an index (integer) in a token (str) using the vocab.'
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        if False:
            return 10
        'Converts a sequence of tokens (string) in a single string.'
        current_sub_tokens = []
        out_string = ''
        prev_is_special = False
        for token in tokens:
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += ' '
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            for i in range(10):
                print('nop')
        if not os.path.isdir(save_directory):
            logger.error(f'Vocabulary path ({save_directory}) should be a directory')
            return
        out_vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, 'wb') as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        return (out_vocab_file,)

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            i = 10
            return i + 15
        '\n        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding\n        special tokens using the tokenizer `prepare_for_model` method.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n            already_has_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not the token list is already formatted with special tokens for the model.\n\n        Returns:\n            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.\n        '
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + [0] * len(token_ids_0) + suffix_ones
        return prefix_ones + [0] * len(token_ids_0) + [0] * len(token_ids_1) + suffix_ones

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. An MBART-50 sequence has the following format, where `X` represents the sequence:\n\n        - `input_ids` (for encoder) `[src_lang_code] X [eos]`\n        - `labels`: (for decoder) `[tgt_lang_code] X [eos]`\n\n        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a\n        separator.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def _build_translation_inputs(self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        if False:
            while True:
                i = 10
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        if False:
            print('Hello World!')
        'Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos].'
        self.cur_lang_code_id = self.lang_code_to_id[src_lang]
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        if False:
            return 10
        'Reset the special tokens to the target language setting. prefix=[tgt_lang_code] and suffix=[eos].'
        self.cur_lang_code_id = self.lang_code_to_id[tgt_lang]
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]