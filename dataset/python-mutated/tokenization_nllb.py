import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
from ...utils import logging
logger = logging.get_logger(__name__)
SPIECE_UNDERLINE = '▁'
VOCAB_FILES_NAMES = {'vocab_file': 'sentencepiece.bpe.model'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'facebook/nllb-200-distilled-600M': 'https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/sentencepiece.bpe.model'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'facebook/nllb-200-distilled-600M': 1024}
FAIRSEQ_LANGUAGE_CODES = ['ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'pes_Arab', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'knc_Arab', 'knc_Latn', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kon_Latn', 'kor_Hang', 'kmr_Latn', 'lao_Laoo', 'lvs_Latn', 'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Latn', 'mkd_Cyrl', 'plt_Latn', 'mlt_Latn', 'mni_Beng', 'khk_Cyrl', 'mos_Latn', 'mri_Latn', 'zsm_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'gaz_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'pbt_Arab', 'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Beng', 'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'als_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 'taq_Latn', 'taq_Tfng', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zul_Latn']

class NllbTokenizer(PreTrainedTokenizer):
    """
    Construct an NLLB tokenizer.

    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.

    Examples:

    ```python
    >>> from transformers import NllbTokenizer

    >>> tokenizer = NllbTokenizer.from_pretrained(
    ...     "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
    ... )
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
    ```

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

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
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        src_lang (`str`, *optional*):
            The language to use as source language for translation.
        tgt_lang (`str`, *optional*):
            The language to use as target language for translation.
        sp_model_kwargs (`Dict[str, str]`):
            Additional keyword arguments to pass to the model initialization.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ['input_ids', 'attention_mask']
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(self, vocab_file, bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>', tokenizer_file=None, src_lang=None, tgt_lang=None, sp_model_kwargs: Optional[Dict[str, Any]]=None, additional_special_tokens=None, legacy_behaviour=False, **kwargs):
        if False:
            i = 10
            return i + 15
        mask_token = AddedToken(mask_token, normalized=True, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.legacy_behaviour = legacy_behaviour
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
        self._src_lang = src_lang if src_lang is not None else 'eng_Latn'
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        _additional_special_tokens = list(self.lang_code_to_id.keys())
        if additional_special_tokens is not None:
            _additional_special_tokens.extend([t for t in additional_special_tokens if t not in _additional_special_tokens])
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, sep_token=sep_token, cls_token=cls_token, pad_token=pad_token, mask_token=mask_token, tokenizer_file=tokenizer_file, src_lang=src_lang, tgt_lang=tgt_lang, additional_special_tokens=_additional_special_tokens, sp_model_kwargs=self.sp_model_kwargs, legacy_behaviour=legacy_behaviour, **kwargs)
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def __getstate__(self):
        if False:
            return 10
        state = self.__dict__.copy()
        state['sp_model'] = None
        state['sp_model_proto'] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        if False:
            return 10
        self.__dict__ = d
        if not hasattr(self, 'sp_model_kwargs'):
            self.sp_model_kwargs = {}
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        if False:
            while True:
                i = 10
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

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            return 10
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
            for i in range(10):
                print('nop')
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. An NLLB sequence has the following format, where `X` represents the sequence:\n\n        - `input_ids` (for encoder) `X [eos, src_lang_code]`\n        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`\n\n        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a\n        separator.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. nllb does not\n        make use of token type ids, therefore a list of zeros is returned.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of zeros.\n\n        '
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

    def get_vocab(self):
        if False:
            return 10
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        if False:
            while True:
                i = 10
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        if False:
            return 10
        'Converts a token (str) in an id using the vocab.'
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        if False:
            while True:
                i = 10
        'Converts an index (integer) in a token (str) using the vocab.'
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        if False:
            for i in range(10):
                print('nop')
        'Converts a sequence of tokens (strings for sub-words) in a single string.'
        out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            return 10
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

    def prepare_seq2seq_batch(self, src_texts: List[str], src_lang: str='eng_Latn', tgt_texts: Optional[List[str]]=None, tgt_lang: str='fra_Latn', **kwargs) -> BatchEncoding:
        if False:
            while True:
                i = 10
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        if False:
            i = 10
            return i + 15
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        if False:
            print('Hello World!')
        'Reset the special tokens to the source lang setting.\n        - In legacy mode: No prefix and suffix=[eos, src_lang_code].\n        - In default mode: Prefix=[src_lang_code], suffix = [eos]\n        '
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        if False:
            return 10
        'Reset the special tokens to the target lang setting.\n        - In legacy mode: No prefix and suffix=[eos, tgt_lang_code].\n        - In default mode: Prefix=[tgt_lang_code], suffix = [eos]\n        '
        self.cur_lang_code = self.lang_code_to_id[lang]
        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]