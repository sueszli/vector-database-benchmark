"""Tokenization class for Wav2Vec2Phoneme."""
import json
import os
import sys
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import ModelOutput, is_flax_available, is_tf_available, is_torch_available, logging, requires_backends, to_py_obj
logger = logging.get_logger(__name__)
if TYPE_CHECKING:
    if is_torch_available():
        import torch
    if is_tf_available():
        import tensorflow as tf
    if is_flax_available():
        import jax.numpy as jnp
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'tokenizer_config_file': 'tokenizer_config.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'facebook/wav2vec2-lv-60-espeak-cv-ft': 'https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/resolve/main/vocab.json'}, 'tokenizer_config_file': {'facebook/wav2vec2-lv-60-espeak-cv-ft': 'https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/resolve/main/tokenizer_config.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'facebook/wav2vec2-lv-60-espeak-cv-ft': sys.maxsize}
ListOfDict = List[Dict[str, Union[int, str]]]

@dataclass
class Wav2Vec2PhonemeCTCTokenizerOutput(ModelOutput):
    """
    Output type of [` Wav2Vec2PhonemeCTCTokenizer`], with transcription.

    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        char_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`):
            Offsets of the decoded characters. In combination with sampling rate and model downsampling rate char
            offsets can be used to compute time stamps for each charater. Total logit score of the beam associated with
            produced text.
    """
    text: Union[List[str], str]
    char_offsets: Union[List[ListOfDict], ListOfDict] = None

class Wav2Vec2PhonemeCTCTokenizer(PreTrainedTokenizer):
    """
    Constructs a Wav2Vec2PhonemeCTC tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        do_phonemize (`bool`, *optional*, defaults to `True`):
            Whether the tokenizer should phonetize the input or not. Only if a sequence of phonemes is passed to the
            tokenizer, `do_phonemize` should be set to `False`.
        phonemizer_lang (`str`, *optional*, defaults to `"en-us"`):
            The language of the phoneme set to which the tokenizer should phonetize the input text to.
        phonemizer_backend (`str`, *optional*. defaults to `"espeak"`):
            The backend phonetization library that shall be used by the phonemizer library. Defaults to `espeak-ng`.
            See the [phonemizer package](https://github.com/bootphon/phonemizer#readme). for more information.

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, vocab_file, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', phone_delimiter_token=' ', word_delimiter_token=None, do_phonemize=True, phonemizer_lang='en-us', phonemizer_backend='espeak', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._word_delimiter_token = word_delimiter_token
        self._phone_delimiter_token = phone_delimiter_token
        self.do_phonemize = do_phonemize
        self.phonemizer_lang = phonemizer_lang
        self.phonemizer_backend = phonemizer_backend
        if do_phonemize:
            self.init_backend(self.phonemizer_lang)
        with open(vocab_file, encoding='utf-8') as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for (k, v) in self.encoder.items()}
        super().__init__(unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, word_delimiter_token=word_delimiter_token, phone_delimiter_token=phone_delimiter_token, do_phonemize=do_phonemize, phonemizer_lang=phonemizer_lang, phonemizer_backend=phonemizer_backend, **kwargs)

    @property
    def vocab_size(self) -> int:
        if False:
            return 10
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        if False:
            i = 10
            return i + 15
        vocab = dict(self.encoder.copy())
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool=False) -> int:
        if False:
            print('Hello World!')
        to_add = []
        for token in new_tokens:
            if isinstance(token, str):
                to_add.append(AddedToken(token, rstrip=False, lstrip=False, normalized=True, special=special_tokens))
            else:
                to_add.append(token)
        return super()._add_tokens(to_add, special_tokens)

    def init_backend(self, phonemizer_lang: str):
        if False:
            print('Hello World!')
        '\n        Initializes the backend.\n\n        Args:\n            phonemizer_lang (`str`): The language to be used.\n        '
        requires_backends(self, 'phonemizer')
        from phonemizer.backend import BACKENDS
        self.backend = BACKENDS[self.phonemizer_backend](phonemizer_lang, language_switch='remove-flags')

    def prepare_for_tokenization(self, text: str, is_split_into_words: bool=False, phonemizer_lang: Optional[str]=None, do_phonemize: Optional[bool]=None) -> Tuple[str, Dict[str, Any]]:
        if False:
            while True:
                i = 10
        '\n        Performs any necessary transformations before tokenization.\n\n        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the\n        `kwargs` at the end of the encoding process to be sure all the arguments have been used.\n\n        Args:\n            text (`str`):\n                The text to prepare.\n            is_split_into_words (`bool`, *optional*, defaults to `False`):\n                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the\n                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)\n                which it will tokenize. This is useful for NER or token classification.\n            phonemizer_lang (`str`, *optional*):\n                The language of the phoneme set to which the tokenizer should phonetize the input text to.\n            do_phonemize (`bool`, *optional*):\n                Whether the tokenizer should phonetize the input text or not. Only if a sequence of phonemes is passed\n                to the tokenizer, `do_phonemize` should be set to `False`.\n\n\n        Returns:\n            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.\n        '
        if is_split_into_words:
            text = ' ' + text
        if do_phonemize is not None:
            self.do_phonemize = do_phonemize
        if phonemizer_lang is not None:
            self.phonemizer_lang = phonemizer_lang
            self.init_backend(phonemizer_lang)
        return (text, {})

    def _tokenize(self, text, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts a string in a sequence of tokens (string), using the tokenizer.\n        '
        text = text.strip()
        if self.do_phonemize:
            text = text.lower()
            text = self.phonemize(text, self.phonemizer_lang)
        tokens = text.split(' ')
        tokens = list(filter(lambda p: p.strip() != '', tokens))
        return tokens

    def phonemize(self, text: str, phonemizer_lang: Optional[str]=None) -> str:
        if False:
            while True:
                i = 10
        from phonemizer.separator import Separator
        word_delimiter = self.word_delimiter_token + ' ' if self.word_delimiter_token is not None else ''
        if phonemizer_lang is not None and phonemizer_lang != self.phonemizer_lang:
            self.init_backend(phonemizer_lang)
        else:
            phonemizer_lang = self.phonemizer_lang
        separator = Separator(phone=self.phone_delimiter_token, word=word_delimiter, syllable='')
        phonemes = self.backend.phonemize([text], separator=separator)
        phonemes = phonemes[0].strip()
        return phonemes

    @property
    def word_delimiter_token(self) -> str:
        if False:
            return 10
        '\n        `str`: Word delimiter token. Log an error if used while not having been set.\n        '
        if self._word_delimiter_token is None:
            if self.verbose:
                logger.error('Using word_delimiter_token, but it is not set yet.')
            return None
        return str(self._word_delimiter_token)

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        if False:
            print('Hello World!')
        '\n        `Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns `None` if the token has not been\n        set.\n        '
        if self._word_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.word_delimiter_token)

    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        if False:
            print('Hello World!')
        self._word_delimiter_token = value

    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        if False:
            while True:
                i = 10
        self._word_delimiter_token = self.convert_tokens_to_ids(value)

    @property
    def phone_delimiter_token(self) -> str:
        if False:
            print('Hello World!')
        '\n        `str`: Word delimiter token. Log an error if used while not having been set.\n        '
        if self._phone_delimiter_token is None:
            if self.verbose:
                logger.error('Using phone_delimiter_token, but it is not set yet.')
            return None
        return str(self._phone_delimiter_token)

    @property
    def phone_delimiter_token_id(self) -> Optional[int]:
        if False:
            print('Hello World!')
        '\n        `Optional[int]`: Id of the phone_delimiter_token in the vocabulary. Returns `None` if the token has not been\n        set.\n        '
        if self._phone_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.phone_delimiter_token)

    @phone_delimiter_token.setter
    def phone_delimiter_token(self, value):
        if False:
            i = 10
            return i + 15
        self._phone_delimiter_token = value

    @phone_delimiter_token_id.setter
    def phone_delimiter_token_id(self, value):
        if False:
            return 10
        self._phone_delimiter_token = self.convert_tokens_to_ids(value)

    def _convert_token_to_id(self, token: str) -> int:
        if False:
            print('Hello World!')
        'Converts a token (str) in an index (integer) using the vocab.'
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        if False:
            while True:
                i = 10
        'Converts an index (integer) in a token (str) using the vocab.'
        result = self.decoder.get(index, self.unk_token)
        return result

    def convert_tokens_to_string(self, tokens: List[str], group_tokens: bool=True, spaces_between_special_tokens: bool=False, filter_word_delimiter_token: bool=True, output_char_offsets: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.\n        '
        if group_tokens:
            (chars, char_repetitions) = zip(*((token, len(list(group_iter))) for (token, group_iter) in groupby(tokens)))
        else:
            chars = tokens
            char_repetitions = len(tokens) * [1]
        processed_chars = list(filter(lambda char: char != self.pad_token, chars))
        if filter_word_delimiter_token and self.word_delimiter_token is not None:
            processed_chars = list(filter(lambda token: token != self.word_delimiter_token, processed_chars))
        char_offsets = None
        if output_char_offsets:
            word_delimiter_token_for_offsets = self.word_delimiter_token if filter_word_delimiter_token is True else None
            char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token, word_delimiter_token=word_delimiter_token_for_offsets)
            if len(char_offsets) != len(processed_chars):
                raise ValueError(f'`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars} have to be of the same length, but are: `len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`: {len(processed_chars)}')
            for (i, char) in enumerate(processed_chars):
                char_offsets[i]['char'] = char
        string = ' '.join(processed_chars).strip()
        return {'text': string, 'char_offsets': char_offsets}

    @staticmethod
    def _compute_offsets(char_repetitions: List[int], chars: List[str], ctc_token: int, word_delimiter_token: Optional[int]=None) -> List[Dict[str, Union[str, int]]]:
        if False:
            return 10
        end_indices = np.asarray(char_repetitions).cumsum()
        start_indices = np.concatenate(([0], end_indices[:-1]))
        offsets = [{'char': t, 'start_offset': s, 'end_offset': e} for (t, s, e) in zip(chars, start_indices, end_indices)]
        offsets = list(filter(lambda offsets: offsets['char'] != ctc_token, offsets))
        if word_delimiter_token is not None:
            offsets = list(filter(lambda offsets: offsets['char'] != word_delimiter_token, offsets))
        return offsets

    def _decode(self, token_ids: List[int], skip_special_tokens: bool=False, clean_up_tokenization_spaces: bool=None, group_tokens: bool=True, filter_word_delimiter_token: bool=True, spaces_between_special_tokens: bool=False, output_char_offsets: bool=False) -> str:
        if False:
            while True:
                i = 10
        '\n        special _decode function is needed for Wav2Vec2PhonemeTokenizer because added tokens should be treated exactly\n        the same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be\n        called on the whole token list and not individually on added tokens\n        '
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)
        string_output = self.convert_tokens_to_string(result, group_tokens=group_tokens, spaces_between_special_tokens=spaces_between_special_tokens, filter_word_delimiter_token=filter_word_delimiter_token, output_char_offsets=output_char_offsets)
        text = string_output['text']
        clean_up_tokenization_spaces = clean_up_tokenization_spaces if clean_up_tokenization_spaces is not None else self.clean_up_tokenization_spaces
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)
        if output_char_offsets:
            return Wav2Vec2PhonemeCTCTokenizerOutput(text=text, char_offsets=string_output['char_offsets'])
        else:
            return text

    def decode(self, token_ids: Union[int, List[int], 'np.ndarray', 'torch.Tensor', 'tf.Tensor'], skip_special_tokens: bool=False, clean_up_tokenization_spaces: bool=None, output_char_offsets: bool=False, **kwargs) -> str:
        if False:
            print('Hello World!')
        '\n        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special\n        tokens and clean up tokenization spaces.\n\n        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.\n\n        Args:\n            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):\n                List of tokenized input ids. Can be obtained using the `__call__` method.\n            skip_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not to remove special tokens in the decoding.\n            clean_up_tokenization_spaces (`bool`, *optional*):\n                Whether or not to clean up the tokenization spaces.\n            output_char_offsets (`bool`, *optional*, defaults to `False`):\n                Whether or not to output character offsets. Character offsets can be used in combination with the\n                sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.\n\n                <Tip>\n\n                Please take a look at the Example of [`~models.wav2vec2.tokenization_wav2vec2.decode`] to better\n                understand how to make use of `output_word_offsets`.\n                [`~model.wav2vec2_phoneme.tokenization_wav2vec2_phoneme.batch_decode`] works the same way with\n                phonemes.\n\n                </Tip>\n\n            kwargs (additional keyword arguments, *optional*):\n                Will be passed to the underlying model specific decode method.\n\n        Returns:\n            `str` or [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`]: The decoded\n            sentence. Will be a [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`]\n            when `output_char_offsets == True`.\n        '
        token_ids = to_py_obj(token_ids)
        return self._decode(token_ids=token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces, output_char_offsets=output_char_offsets, **kwargs)

    def batch_decode(self, sequences: Union[List[int], List[List[int]], 'np.ndarray', 'torch.Tensor', 'tf.Tensor'], skip_special_tokens: bool=False, clean_up_tokenization_spaces: bool=None, output_char_offsets: bool=False, **kwargs) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Convert a list of lists of token ids into a list of strings by calling decode.\n\n        Args:\n            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):\n                List of tokenized input ids. Can be obtained using the `__call__` method.\n            skip_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not to remove special tokens in the decoding.\n            clean_up_tokenization_spaces (`bool`, *optional*):\n                Whether or not to clean up the tokenization spaces.\n            output_char_offsets (`bool`, *optional*, defaults to `False`):\n                Whether or not to output character offsets. Character offsets can be used in combination with the\n                sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.\n\n                <Tip>\n\n                Please take a look at the Example of [`~models.wav2vec2.tokenization_wav2vec2.decode`] to better\n                understand how to make use of `output_word_offsets`.\n                [`~model.wav2vec2_phoneme.tokenization_wav2vec2_phoneme.batch_decode`] works analogous with phonemes\n                and batched output.\n\n                </Tip>\n\n            kwargs (additional keyword arguments, *optional*):\n                Will be passed to the underlying model specific decode method.\n\n        Returns:\n            `List[str]` or [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`]: The\n            decoded sentence. Will be a\n            [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`] when\n            `output_char_offsets == True`.\n        '
        batch_decoded = [self.decode(seq, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces, output_char_offsets=output_char_offsets, **kwargs) for seq in sequences]
        if output_char_offsets:
            return Wav2Vec2PhonemeCTCTokenizerOutput({k: [d[k] for d in batch_decoded] for k in batch_decoded[0]})
        return batch_decoded

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            while True:
                i = 10
        if not os.path.isdir(save_directory):
            logger.error(f'Vocabulary path ({save_directory}) should be a directory')
            return
        vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
        return (vocab_file,)