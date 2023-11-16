"""
 Tokenization classes for fast tokenizers (provided by HuggingFace's tokenizers library). For slow (python) tokenizers
 see tokenization_utils.py
"""
import copy
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import tokenizers.pre_tokenizers as pre_tokenizers_fast
from tokenizers import Encoding as EncodingFast
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from .convert_slow_tokenizer import convert_slow_tokenizer
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import INIT_TOKENIZER_DOCSTRING, AddedToken, BatchEncoding, PreTokenizedInput, PreTokenizedInputPair, PreTrainedTokenizerBase, SpecialTokensMixin, TextInput, TextInputPair, TruncationStrategy
from .utils import PaddingStrategy, add_end_docstrings, logging
logger = logging.get_logger(__name__)
TOKENIZER_FILE = 'tokenizer.json'
SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
TOKENIZER_CONFIG_FILE = 'tokenizer_config.json'
ADDED_TOKENS_FILE = 'added_tokens.json'
INIT_TOKENIZER_DOCSTRING += '\n        tokenizer_object ([`tokenizers.Tokenizer`]):\n            A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗\n            tokenizers](../fast_tokenizers) for more information.\n        tokenizer_file ([`str`]):\n            A path to a local JSON file representing a previously serialized [`tokenizers.Tokenizer`] object from 🤗\n            tokenizers.\n'
MODEL_TO_TRAINER_MAPPING = {'BPE': BpeTrainer, 'Unigram': UnigramTrainer, 'WordLevel': WordLevelTrainer, 'WordPiece': WordPieceTrainer}
VOCAB_FILES_NAMES = {'tokenizer_file': TOKENIZER_FILE}

@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """
    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class: PreTrainedTokenizer = None

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        tokenizer_object = kwargs.pop('tokenizer_object', None)
        slow_tokenizer = kwargs.pop('__slow_tokenizer', None)
        fast_tokenizer_file = kwargs.pop('tokenizer_file', None)
        from_slow = kwargs.pop('from_slow', False)
        added_tokens_decoder = kwargs.pop('added_tokens_decoder', {})
        if from_slow and slow_tokenizer is None and (self.slow_tokenizer_class is None):
            raise ValueError("Cannot instantiate this tokenizer from a slow version. If it's based on sentencepiece, make sure you have sentencepiece installed.")
        if tokenizer_object is not None:
            fast_tokenizer = copy.deepcopy(tokenizer_object)
        elif fast_tokenizer_file is not None and (not from_slow):
            fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
        elif slow_tokenizer is not None:
            fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        elif self.slow_tokenizer_class is not None:
            slow_tokenizer = self.slow_tokenizer_class(*args, **kwargs)
            fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        else:
            raise ValueError("Couldn't instantiate the backend tokenizer from one of: \n(1) a `tokenizers` library serialization file, \n(2) a slow tokenizer instance to convert or \n(3) an equivalent slow tokenizer class to instantiate and convert. \nYou need to have sentencepiece installed to convert a slow tokenizer to a fast one.")
        self._tokenizer = fast_tokenizer
        if slow_tokenizer is not None:
            kwargs.update(slow_tokenizer.init_kwargs)
        self._decode_use_source_tokenizer = False
        _truncation = self._tokenizer.truncation
        if _truncation is not None:
            self._tokenizer.enable_truncation(**_truncation)
            kwargs.setdefault('max_length', _truncation['max_length'])
            kwargs.setdefault('truncation_side', _truncation['direction'])
            kwargs.setdefault('stride', _truncation['stride'])
            kwargs.setdefault('truncation_strategy', _truncation['strategy'])
        else:
            self._tokenizer.no_truncation()
        _padding = self._tokenizer.padding
        if _padding is not None:
            self._tokenizer.enable_padding(**_padding)
            kwargs.setdefault('pad_token', _padding['pad_token'])
            kwargs.setdefault('pad_token_type_id', _padding['pad_type_id'])
            kwargs.setdefault('padding_side', _padding['direction'])
            kwargs.setdefault('max_length', _padding['length'])
            kwargs.setdefault('pad_to_multiple_of', _padding['pad_to_multiple_of'])
        super().__init__(**kwargs)
        tokens_to_add = [token for (index, token) in sorted(added_tokens_decoder.items(), key=lambda x: x[0]) if token not in self.added_tokens_decoder]
        encoder = list(self.added_tokens_encoder.keys()) + [str(token) for token in tokens_to_add]
        tokens_to_add += [token for token in self.all_special_tokens_extended if token not in encoder and token not in tokens_to_add]
        if len(tokens_to_add) > 0:
            is_last_special = None
            tokens = []
            special_tokens = self.all_special_tokens
            for token in tokens_to_add:
                is_special = token.special or str(token) in special_tokens if isinstance(token, AddedToken) else str(token) in special_tokens
                if is_last_special is None or is_last_special == is_special:
                    tokens.append(token)
                else:
                    self._add_tokens(tokens, special_tokens=is_last_special)
                    tokens = [token]
                is_last_special = is_special
            if tokens:
                self._add_tokens(tokens, special_tokens=is_last_special)

    @property
    def is_fast(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    @property
    def can_save_slow_tokenizer(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        `bool`: Whether or not the slow tokenizer can be saved. Usually for sentencepiece based slow tokenizer, this\n        can only be `True` if the original `"sentencepiece.model"` was not deleted.\n        '
        return True

    @property
    def vocab_size(self) -> int:
        if False:
            print('Hello World!')
        '\n        `int`: Size of the base vocabulary (without the added tokens).\n        '
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> Dict[str, int]:
        if False:
            print('Hello World!')
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab(self) -> Dict[str, int]:
        if False:
            return 10
        return self.get_vocab()

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        if False:
            print('Hello World!')
        '\n        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance\n        optimisation in `self._added_tokens_encoder` for the slow tokenizers.\n        '
        return {k.content: v for (v, k) in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        if False:
            return 10
        '\n        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.\n\n        Returns:\n            `Dict[str, int]`: The added tokens.\n        '
        return self._tokenizer.get_added_tokens_decoder()

    def get_added_vocab(self) -> Dict[str, int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the added tokens in the vocabulary as a dictionary of token to index.\n\n        Returns:\n            `Dict[str, int]`: The added tokens.\n        '
        return {k.content: v for (v, k) in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Size of the full vocabulary with the added tokens.\n        '
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> TokenizerFast:
        if False:
            return 10
        '\n        `tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.\n        '
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        if False:
            for i in range(10):
                print('nop')
        '\n        `tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.\n        '
        return self._tokenizer.decoder

    def _convert_encoding(self, encoding: EncodingFast, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True) -> Tuple[Dict[str, Any], List[EncodingFast]]:
        if False:
            i = 10
            return i + 15
        '\n        Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict and a list\n        of encodings, take care of building a batch from overflowing tokens.\n\n        Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are\n        lists (overflows) of lists (tokens).\n\n        Output shape: (overflows, sequence length)\n        '
        if return_token_type_ids is None:
            return_token_type_ids = 'token_type_ids' in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = 'attention_mask' in self.model_input_names
        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]
        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict['input_ids'].append(e.ids)
            if return_token_type_ids:
                encoding_dict['token_type_ids'].append(e.type_ids)
            if return_attention_mask:
                encoding_dict['attention_mask'].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict['special_tokens_mask'].append(e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict['offset_mapping'].append(e.offsets)
            if return_length:
                encoding_dict['length'].append(len(e.ids))
        return (encoding_dict, encodings)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if False:
            return 10
        '\n        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the\n        vocabulary.\n\n        Args:\n            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).\n\n        Returns:\n            `int` or `List[int]`: The token id or list of token ids.\n        '
        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)
        return [self._convert_token_to_id_with_added_voc(token) for token in tokens]

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._tokenizer.id_to_token(int(index))

    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        if False:
            for i in range(10):
                print('nop')
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)
        return self._tokenizer.add_tokens(new_tokens)

    def num_special_tokens_to_add(self, pair: bool=False) -> int:
        if False:
            print('Hello World!')
        '\n        Returns the number of added tokens when encoding a sequence with special tokens.\n\n        <Tip>\n\n        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put\n        this inside your training loop.\n\n        </Tip>\n\n        Args:\n            pair (`bool`, *optional*, defaults to `False`):\n                Whether the number of added tokens should be computed in the case of a sequence pair or a single\n                sequence.\n\n        Returns:\n            `int`: Number of special tokens added to sequences.\n        '
        return self._tokenizer.num_special_tokens_to_add(pair)

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool=False) -> Union[str, List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and\n        added tokens.\n\n        Args:\n            ids (`int` or `List[int]`):\n                The token id (or token ids) to convert to tokens.\n            skip_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not to remove special tokens in the decoding.\n\n        Returns:\n            `str` or `List[str]`: The decoded token(s).\n        '
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._tokenizer.id_to_token(index))
        return tokens

    def tokenize(self, text: str, pair: Optional[str]=None, add_special_tokens: bool=False, **kwargs) -> List[str]:
        if False:
            i = 10
            return i + 15
        return self.encode_plus(text=text, text_pair=pair, add_special_tokens=add_special_tokens, **kwargs).tokens()

    def set_truncation_and_padding(self, padding_strategy: PaddingStrategy, truncation_strategy: TruncationStrategy, max_length: int, stride: int, pad_to_multiple_of: Optional[int]):
        if False:
            while True:
                i = 10
        '\n        Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers\n        library) and restore the tokenizer settings afterwards.\n\n        The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a\n        padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed\n        section.\n\n        Args:\n            padding_strategy ([`~utils.PaddingStrategy`]):\n                The kind of padding that will be applied to the input\n            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`]):\n                The kind of truncation that will be applied to the input\n            max_length (`int`):\n                The maximum size of a sequence.\n            stride (`int`):\n                The stride to use when handling overflow.\n            pad_to_multiple_of (`int`, *optional*):\n                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable\n                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).\n        '
        _truncation = self._tokenizer.truncation
        _padding = self._tokenizer.padding
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            if _truncation is not None:
                self._tokenizer.no_truncation()
        else:
            target = {'max_length': max_length, 'stride': stride, 'strategy': truncation_strategy.value, 'direction': self.truncation_side}
            if _truncation is None:
                current = None
            else:
                current = {k: _truncation.get(k, None) for k in target}
            if current != target:
                self._tokenizer.enable_truncation(**target)
        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            if _padding is not None:
                self._tokenizer.no_padding()
        else:
            length = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
            target = {'length': length, 'direction': self.padding_side, 'pad_id': self.pad_token_id, 'pad_token': self.pad_token, 'pad_type_id': self.pad_token_type_id, 'pad_to_multiple_of': pad_to_multiple_of}
            if _padding != target:
                self._tokenizer.enable_padding(**target)

    def _batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]], add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, is_split_into_words: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[str]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True) -> BatchEncoding:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(batch_text_or_text_pairs, (tuple, list)):
            raise TypeError(f'batch_text_or_text_pairs has to be a list or a tuple (got {type(batch_text_or_text_pairs)})')
        self.set_truncation_and_padding(padding_strategy=padding_strategy, truncation_strategy=truncation_strategy, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of)
        encodings = self._tokenizer.encode_batch(batch_text_or_text_pairs, add_special_tokens=add_special_tokens, is_pretokenized=is_split_into_words)
        tokens_and_encodings = [self._convert_encoding(encoding=encoding, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose) for encoding in encodings]
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0].keys():
            stack = [e for (item, _) in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
        sanitized_encodings = [e for (_, item) in tokens_and_encodings for e in item]
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for (i, (toks, _)) in enumerate(tokens_and_encodings):
                overflow_to_sample_mapping += [i] * len(toks['input_ids'])
            sanitized_tokens['overflow_to_sample_mapping'] = overflow_to_sample_mapping
        for input_ids in sanitized_tokens['input_ids']:
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
        return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

    def _encode_plus(self, text: Union[TextInput, PreTokenizedInput], text_pair: Optional[Union[TextInput, PreTokenizedInput]]=None, add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, is_split_into_words: bool=False, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[bool]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        if False:
            for i in range(10):
                print('nop')
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_output = self._batch_encode_plus(batched_input, is_split_into_words=is_split_into_words, add_special_tokens=add_special_tokens, padding_strategy=padding_strategy, truncation_strategy=truncation_strategy, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)
        if return_tensors is None and (not return_overflowing_tokens):
            batched_output = BatchEncoding({key: value[0] if len(value) > 0 and isinstance(value[0], list) else value for (key, value) in batched_output.items()}, batched_output.encodings)
        self._eventual_warn_about_too_long_sequence(batched_output['input_ids'], max_length, verbose)
        return batched_output

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.backend_tokenizer.decoder.decode(tokens)

    def _decode(self, token_ids: Union[int, List[int]], skip_special_tokens: bool=False, clean_up_tokenization_spaces: bool=None, **kwargs) -> str:
        if False:
            while True:
                i = 10
        self._decode_use_source_tokenizer = kwargs.pop('use_source_tokenizer', False)
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        clean_up_tokenization_spaces = clean_up_tokenization_spaces if clean_up_tokenization_spaces is not None else self.clean_up_tokenization_spaces
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def _save_pretrained(self, save_directory: Union[str, os.PathLike], file_names: Tuple[str], legacy_format: Optional[bool]=None, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            print('Hello World!')
        '\n        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens as well as in a unique JSON\n        file containing {config + vocab + added-tokens}.\n        '
        save_directory = str(save_directory)
        if self.slow_tokenizer_class is None and legacy_format is True:
            raise ValueError('Your tokenizer does not have a legacy version defined and therefore cannot register this version. You might consider leaving the legacy_format at `None` or setting it to `False`.')
        save_slow = (legacy_format is None or legacy_format is True) and self.slow_tokenizer_class is not None and self.can_save_slow_tokenizer
        save_fast = legacy_format is None or legacy_format is False
        if save_slow:
            added_tokens_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + ADDED_TOKENS_FILE)
            added_vocab = {tok: index for (tok, index) in self.added_tokens_encoder.items() if index >= self.vocab_size}
            if added_vocab:
                with open(added_tokens_file, 'w', encoding='utf-8') as f:
                    out_str = json.dumps(added_vocab, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
                    f.write(out_str)
            vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)
            file_names = file_names + vocab_files + (added_tokens_file,)
        if save_fast:
            tokenizer_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + TOKENIZER_FILE)
            self.backend_tokenizer.save(tokenizer_file)
            file_names = file_names + (tokenizer_file,)
        return file_names

    def train_new_from_iterator(self, text_iterator, vocab_size, length=None, new_special_tokens=None, special_tokens_map=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)\n        as the current one.\n\n        Args:\n            text_iterator (generator of `List[str]`):\n                The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts\n                if you have everything in memory.\n            vocab_size (`int`):\n                The size of the vocabulary you want for your tokenizer.\n            length (`int`, *optional*):\n                The total number of sequences in the iterator. This is used to provide meaningful progress tracking\n            new_special_tokens (list of `str` or `AddedToken`, *optional*):\n                A list of new special tokens to add to the tokenizer you are training.\n            special_tokens_map (`Dict[str, str]`, *optional*):\n                If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special\n                token name to new special token name in this argument.\n            kwargs (`Dict[str, Any]`, *optional*):\n                Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.\n\n        Returns:\n            [`PreTrainedTokenizerFast`]: A new tokenizer of the same type as the original one, trained on\n            `text_iterator`.\n\n        '
        tokenizer_json = json.loads(self._tokenizer.to_str())
        added_tokens = tokenizer_json.pop('added_tokens')
        post_processor = tokenizer_json.pop('post_processor')
        unk_token = None
        if tokenizer_json['model']['type'] == 'BPE':
            tokenizer_json['model']['vocab'] = {}
            tokenizer_json['model']['merges'] = []
        elif tokenizer_json['model']['type'] == 'Unigram':
            if tokenizer_json['model']['unk_id'] is not None:
                unk_id = tokenizer_json['model']['unk_id']
                unk_token = tokenizer_json['model']['vocab'][unk_id][0]
                if special_tokens_map is not None and unk_token in special_tokens_map:
                    unk_token = special_tokens_map[unk_token]
                tokenizer_json['model']['unk_id'] = 0
                tokenizer_json['model']['vocab'] = [[unk_token, 0.0]]
        elif tokenizer_json['model']['type'] in ['WordLevel', 'WordPiece']:
            tokenizer_json['model']['vocab'] = {}
        else:
            raise ValueError(f"This method does not support this type of tokenizer (found {tokenizer_json['model']['type']}) only BPE, Unigram, WordLevel and WordPiece.")
        if special_tokens_map is not None and 'unk_token' in tokenizer_json['model'] and (tokenizer_json['model']['unk_token'] in special_tokens_map):
            tokenizer_json['model']['unk_token'] = special_tokens_map[tokenizer_json['model']['unk_token']]
        tokenizer = TokenizerFast.from_str(json.dumps(tokenizer_json))
        special_tokens = []
        for added_token in added_tokens:
            special = added_token.pop('special', None)
            _ = added_token.pop('id', None)
            if tokenizer_json['model']['type'] != 'Unigram' and (not special):
                continue
            if special_tokens_map is not None and added_token['content'] in special_tokens_map:
                added_token['content'] = special_tokens_map[added_token['content']]
            special_tokens.append(AddedToken(**added_token))
        if new_special_tokens is not None:
            special_tokens.extend(new_special_tokens)
        if tokenizer_json['model']['type'] == 'BPE' and 'continuing_subword_prefix' not in kwargs and (tokenizer_json['model']['continuing_subword_prefix'] is not None):
            kwargs['continuing_subword_prefix'] = tokenizer_json['model']['continuing_subword_prefix']
        if tokenizer_json['model']['type'] == 'BPE' and 'end_of_word_suffix' not in kwargs and (tokenizer_json['model']['end_of_word_suffix'] is not None):
            kwargs['end_of_word_suffix'] = tokenizer_json['model']['end_of_word_suffix']
        if tokenizer_json['model']['type'] == 'Unigram' and unk_token is not None:
            kwargs['unk_token'] = unk_token
        if tokenizer_json['pre_tokenizer'] is not None and tokenizer_json['pre_tokenizer']['type'] == 'ByteLevel':
            kwargs['initial_alphabet'] = pre_tokenizers_fast.ByteLevel.alphabet()
        trainer_class = MODEL_TO_TRAINER_MAPPING[tokenizer_json['model']['type']]
        trainer = trainer_class(vocab_size=vocab_size, special_tokens=special_tokens, **kwargs)
        tokenizer.train_from_iterator(text_iterator, length=length, trainer=trainer)
        if post_processor is not None:
            trained_tokenizer_json = json.loads(tokenizer.to_str())
            if 'special_tokens' in post_processor:
                for key in post_processor['special_tokens']:
                    tokens = post_processor['special_tokens'][key]['tokens']
                    if special_tokens_map is not None:
                        tokens = [special_tokens_map.get(token, token) for token in tokens]
                    post_processor['special_tokens'][key]['tokens'] = tokens
                    post_processor['special_tokens'][key]['ids'] = [tokenizer.token_to_id(token) for token in tokens]
            for special_token in ['cls', 'sep']:
                if special_token in post_processor:
                    (token, _) = post_processor[special_token]
                    if special_tokens_map is not None and token in special_tokens_map:
                        token = special_tokens_map[token]
                    token_id = tokenizer.token_to_id(token)
                    post_processor[special_token] = [token, token_id]
            trained_tokenizer_json['post_processor'] = post_processor
            tokenizer = TokenizerFast.from_str(json.dumps(trained_tokenizer_json))
        kwargs = self.init_kwargs.copy()
        special_tokens_list = SpecialTokensMixin.SPECIAL_TOKENS_ATTRIBUTES.copy()
        special_tokens_list.remove('additional_special_tokens')
        for token in special_tokens_list:
            if getattr(self, f'_{token}') is not None:
                special_token = getattr(self, token)
                if special_tokens_map is not None and special_token in special_tokens_map:
                    special_token = special_tokens_map[special_token]
                special_token_full = getattr(self, f'_{token}')
                if isinstance(special_token_full, AddedToken):
                    kwargs[token] = AddedToken(special_token, single_word=special_token_full.single_word, lstrip=special_token_full.lstrip, rstrip=special_token_full.rstrip, normalized=special_token_full.normalized, special=True)
                else:
                    kwargs[token] = special_token
        additional_special_tokens = self.additional_special_tokens
        if new_special_tokens is not None:
            additional_special_tokens.extend(new_special_tokens)
        if len(additional_special_tokens) > 0:
            kwargs['additional_special_tokens'] = additional_special_tokens
        return self.__class__(tokenizer_object=tokenizer, **kwargs)