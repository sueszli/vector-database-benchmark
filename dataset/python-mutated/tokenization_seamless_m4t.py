"""Tokenization classes for SeamlessM4T."""
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece as spm
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import BatchEncoding, PreTokenizedInput, PreTrainedTokenizer, TextInput
from ...tokenization_utils_base import AddedToken
from ...utils import PaddingStrategy, logging
logger = logging.get_logger(__name__)
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'facebook/hf-seamless-m4t-medium': 'https://huggingface.co/facebook/hf-seamless-m4t-medium/blob/main/sentencepiece.bpe.model'}}
SPIECE_UNDERLINE = '▁'
VOCAB_FILES_NAMES = {'vocab_file': 'sentencepiece.bpe.model'}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'facebook/hf-seamless-m4t-medium': 2048}

class SeamlessM4TTokenizer(PreTrainedTokenizer):
    """
    Construct a SeamlessM4T tokenizer.

    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    The tokenization method is `<language code> <tokens> <eos>` for source language documents, and `<eos> <language
    code> <tokens> <eos>` for target language documents.

    Examples:

    ```python
    >>> from transformers import SeamlessM4TTokenizer

    >>> tokenizer = SeamlessM4TTokenizer.from_pretrained(
    ...     "facebook/hf-seamless-m4t-medium", src_lang="eng", tgt_lang="fra"
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
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        src_lang (`str`, *optional*, defaults to `"eng"`):
            The language to use as source language for translation.
        tgt_lang (`str`, *optional*, defaults to `"fra"`):
            The language to use as target language for translation.
        sp_model_kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the model initialization.
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            A tuple or a list of additional special tokens. Can be used to specify the list of languages that will be
            supported by the tokenizer.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ['input_ids', 'attention_mask']
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(self, vocab_file, bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', tokenizer_file=None, src_lang='eng', tgt_lang='fra', sp_model_kwargs: Optional[Dict[str, Any]]=None, additional_special_tokens=None, **kwargs):
        if False:
            print('Hello World!')
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.legacy = False
        self.vocab_file = vocab_file
        self.sp_model = self.get_spm_processor(kwargs.pop('from_slow', False))
        self._added_tokens_decoder = {0: AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token, 1: AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token, 2: AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token, 3: AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token}
        self.fairseq_offset = 1
        self.sp_model_size = len(self.sp_model)
        self._src_lang = f'__{src_lang}__' if '__' not in src_lang else src_lang
        self._tgt_lang = f'__{tgt_lang}__' if '__' not in tgt_lang else tgt_lang
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, sep_token=sep_token, cls_token=cls_token, pad_token=pad_token, tokenizer_file=tokenizer_file, src_lang=src_lang, tgt_lang=tgt_lang, additional_special_tokens=additional_special_tokens, sp_model_kwargs=self.sp_model_kwargs, **kwargs)
        self.set_src_lang_special_tokens(self._src_lang)
        self.set_tgt_lang_special_tokens(self._tgt_lang)

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
        return len(self.sp_model)

    def __call__(self, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]=None, text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]=None, text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]=None, text_pair_target: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]=None, padding: Union[bool, str, PaddingStrategy]=True, pad_to_multiple_of: Optional[int]=2, src_lang: Optional[str]=None, tgt_lang: Optional[str]=None, **kwargs):
        if False:
            return 10
        "\n        Args:\n            text (`str`, `List[str]`, `List[List[str]]`, *optional*):\n                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings\n                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set\n                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).\n            text_pair (`str`, `List[str]`, `List[List[str]]`, *optional*):\n                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings\n                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set\n                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).\n            text_target (`str`, `List[str]`, `List[List[str]]`, *optional*):\n                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a\n                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),\n                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).\n            text_pair_target (`str`, `List[str]`, `List[List[str]]`, *optional*):\n                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a\n                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),\n                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).\n            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):\n                 Select a strategy to pad the returned sequences (according to the model's padding side and padding\n                 index) among:\n\n                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single\n                  sequence if provided).\n                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum\n                  acceptable input length for the model if that argument is not provided.\n                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different\n                  lengths).\n            pad_to_multiple_of (`int`, *optional*):\n                If set will pad the sequence to a multiple of the provided value.\n\n                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability\n                `>= 7.5` (Volta).\n            src_lang (`str`, *optional*):\n                A string representing the source language. If not specified, the last `src_lang` specified (either\n                during initialization or when calling this tokenizer) will be used.\n            tgt_lang (`str`, *optional*):\n                A string representing the target language. If not specified, the last `tgt_lang` specified (either\n                during initialization or when calling this tokenizer) will be used.\n            kwargs (*optional*):\n                Remaining dictionary of keyword arguments that will be passed to [`PreTrainedTokenizer.__call__`].\n        "
        if src_lang is not None:
            self.src_lang = src_lang
        if tgt_lang is not None:
            self.tgt_lang = tgt_lang
        output = super().__call__(text=text, text_pair=text_pair, text_target=text_target, text_pair_target=text_pair_target, padding=padding, pad_to_multiple_of=pad_to_multiple_of, **kwargs)
        return BatchEncoding(output, tensor_type=kwargs.get('return_tensors'))

    @property
    def src_lang(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        if False:
            return 10
        if '__' not in new_src_lang:
            self._src_lang = f'__{new_src_lang}__'
        else:
            self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def tgt_lang(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._tgt_lang

    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang: str) -> None:
        if False:
            i = 10
            return i + 15
        if '__' not in new_tgt_lang:
            self._tgt_lang = f'__{new_tgt_lang}__'
        else:
            self._tgt_lang = new_tgt_lang
        self.set_tgt_lang_special_tokens(self._tgt_lang)

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            while True:
                i = 10
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
            while True:
                i = 10
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. An NLLB sequence has the following format, where `X` represents the sequence:\n\n        - `input_ids` (for encoder) `X [eos, src_lang_code]`\n        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`\n\n        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a\n        separator.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. nllb does not\n        make use of token type ids, therefore a list of zeros is returned.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of zeros.\n\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Used by translation pipeline, to prepare inputs for the generate function'
        if src_lang is None or tgt_lang is None:
            raise ValueError('Translation requires a `src_lang` and a `tgt_lang` for this model.')
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        if '__' not in tgt_lang:
            tgt_lang = f'__{tgt_lang}__'
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs['forced_bos_token_id'] = tgt_lang_id
        return inputs

    def get_vocab(self):
        if False:
            while True:
                i = 10
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.fairseq_offset, self.vocab_size + self.fairseq_offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def unk_token_length(self):
        if False:
            print('Hello World!')
        return len(self.sp_model.encode(str(self.unk_token)))

    def get_spm_processor(self, from_slow=False):
        if False:
            print('Hello World!')
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        if self.legacy or from_slow:
            tokenizer.Load(self.vocab_file)
            return tokenizer
        with open(self.vocab_file, 'rb') as f:
            sp_model = f.read()
            model_pb2 = import_protobuf(f'The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)')
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    def tokenize(self, text: 'TextInput', add_special_tokens=False, **kwargs) -> List[str]:
        if False:
            return 10
        '\n        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the\n        first token is special.\n        '
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)
        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, ' '), **kwargs)
        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and (tokens[1] in self.all_special_tokens):
            tokens = tokens[1:]
        return tokens

    def _tokenize(self, text, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a tokenized string.\n\n        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any\n        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give\n        `[\'H\', \'e\', \'y\']` instead of `[\'▁He\', \'y\']`. Thus we always encode `f"{unk_token}text"` and strip the\n        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.\n        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.\n        '
        tokens = self.sp_model.encode(text, out_type=str)
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, ' ')):
            return tokens
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        return tokens[self.unk_token_length:] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        if False:
            for i in range(10):
                print('nop')
        'Converts a token (str) in an id using the vocab.'
        spm_id = self.sp_model.PieceToId(token)
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        if False:
            print('Hello World!')
        'Converts an index (integer) in a token (str) using the vocab.'
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        if False:
            print('Hello World!')
        'Converts a sequence of tokens (strings for sub-words) in a single string.'
        if tokens[0].startswith(SPIECE_UNDERLINE):
            tokens[0] = tokens[0][1:]
        out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            print('Hello World!')
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

    def prepare_seq2seq_batch(self, src_texts: List[str], src_lang: str='eng', tgt_texts: Optional[List[str]]=None, tgt_lang: str='fra', **kwargs) -> BatchEncoding:
        if False:
            while True:
                i = 10
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        if False:
            return 10
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        if False:
            return 10
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        if False:
            while True:
                i = 10
        'Reset the special tokens to the source lang setting.\n        Prefix=[src_lang_code], suffix = [eos]\n        '
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)
        self.init_kwargs['src_lang'] = src_lang
        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(f'`src_lang={src_lang}` has not be found in the vocabulary. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id.')
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        if False:
            i = 10
            return i + 15
        'Reset the special tokens to the target lang setting.\n        Prefix=[eos, tgt_lang_code] and suffix=[eos].\n        '
        self.cur_lang_code = self.convert_tokens_to_ids(lang)
        self.init_kwargs['tgt_lang'] = lang
        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(f'`tgt_lang={lang}` has not be found in the vocabulary. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id.')
        self.prefix_tokens = [self.eos_token_id, self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]