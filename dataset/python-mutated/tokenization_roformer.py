"""Tokenization classes for RoFormer."""
import collections
import os
import unicodedata
from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'junnyu/roformer_chinese_small': 'https://huggingface.co/junnyu/roformer_chinese_small/resolve/main/vocab.txt', 'junnyu/roformer_chinese_base': 'https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/vocab.txt', 'junnyu/roformer_chinese_char_small': 'https://huggingface.co/junnyu/roformer_chinese_char_small/resolve/main/vocab.txt', 'junnyu/roformer_chinese_char_base': 'https://huggingface.co/junnyu/roformer_chinese_char_base/resolve/main/vocab.txt', 'junnyu/roformer_small_discriminator': 'https://huggingface.co/junnyu/roformer_small_discriminator/resolve/main/vocab.txt', 'junnyu/roformer_small_generator': 'https://huggingface.co/junnyu/roformer_small_generator/resolve/main/vocab.txt'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'junnyu/roformer_chinese_small': 1536, 'junnyu/roformer_chinese_base': 1536, 'junnyu/roformer_chinese_char_small': 512, 'junnyu/roformer_chinese_char_base': 512, 'junnyu/roformer_small_discriminator': 128, 'junnyu/roformer_small_generator': 128}
PRETRAINED_INIT_CONFIGURATION = {'junnyu/roformer_chinese_small': {'do_lower_case': True}, 'junnyu/roformer_chinese_base': {'do_lower_case': True}, 'junnyu/roformer_chinese_char_small': {'do_lower_case': True}, 'junnyu/roformer_chinese_char_base': {'do_lower_case': True}, 'junnyu/roformer_small_discriminator': {'do_lower_case': True}, 'junnyu/roformer_small_generator': {'do_lower_case': True}}

def load_vocab(vocab_file):
    if False:
        while True:
            i = 10
    'Loads a vocabulary file into a dictionary.'
    vocab = collections.OrderedDict()
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        tokens = reader.readlines()
    for (index, token) in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab

def whitespace_tokenize(text):
    if False:
        return 10
    'Runs basic whitespace cleaning and splitting on a piece of text.'
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None, do_split_on_punc=True):
        if False:
            print('Hello World!')
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc

    def tokenize(self, text, never_split=None):
        if False:
            while True:
                i = 10
        '\n        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.\n\n        Args:\n            never_split (`List[str]`, *optional*)\n                Kept for backward compatibility purposes. Now implemented directly at the base class level (see\n                [`PreTrainedTokenizer.tokenize`]) List of token not to split.\n        '
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        unicode_normalized_text = unicodedata.normalize('NFC', text)
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))
        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        if False:
            print('Hello World!')
        'Strips accents from a piece of text.'
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text, never_split=None):
        if False:
            for i in range(10):
                print('nop')
        'Splits punctuation on a piece of text.'
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Adds whitespace around any CJK character.'
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        if False:
            for i in range(10):
                print('nop')
        'Checks whether CP is the codepoint of a CJK character.'
        if cp >= 19968 and cp <= 40959 or (cp >= 13312 and cp <= 19903) or (cp >= 131072 and cp <= 173791) or (cp >= 173824 and cp <= 177983) or (cp >= 177984 and cp <= 178207) or (cp >= 178208 and cp <= 183983) or (cp >= 63744 and cp <= 64255) or (cp >= 194560 and cp <= 195103):
            return True
        return False

    def _clean_text(self, text):
        if False:
            print('Hello World!')
        'Performs invalid character removal and whitespace cleanup on text.'
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 65533 or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        if False:
            return 10
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        if False:
            while True:
                i = 10
        '\n        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform\n        tokenization using the given vocabulary.\n\n        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.\n\n        Args:\n            text: A single token or whitespace separated tokens. This should have\n                already been passed through *BasicTokenizer*.\n\n        Returns:\n            A list of wordpiece tokens.\n        '
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

class RoFormerTokenizer(PreTrainedTokenizer):
    """
    Construct a RoFormer tokenizer. Based on [Rust Jieba](https://pypi.org/project/rjieba/).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).

    Example:

    ```python
    >>> from transformers import RoFormerTokenizer

    >>> tokenizer = RoFormerTokenizer.from_pretrained("junnyu/roformer_chinese_base")
    >>> tokenizer.tokenize("今天天气非常好。")
    ['今', '天', '天', '气', '非常', '好', '。']
    ```"""
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None, unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]', tokenize_chinese_chars=True, strip_accents=None, **kwargs):
        if False:
            return 10
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`")
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for (tok, ids) in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=tokenize_chinese_chars, strip_accents=strip_accents)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
        try:
            import rjieba
        except ImportError:
            raise ImportError('You need to install rjieba to use RoFormerTokenizer. See https://pypi.org/project/rjieba/ for installation.')
        self.jieba = rjieba
        super().__init__(do_lower_case=do_lower_case, do_basic_tokenize=do_basic_tokenize, never_split=never_split, unk_token=unk_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, tokenize_chinese_chars=tokenize_chinese_chars, strip_accents=strip_accents, **kwargs)

    @property
    def do_lower_case(self):
        if False:
            for i in range(10):
                print('nop')
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.vocab)

    def __getstate__(self):
        if False:
            return 10
        state = self.__dict__.copy()
        state['jieba'] = None
        return state

    def __setstate__(self, d):
        if False:
            for i in range(10):
                print('nop')
        self.__dict__ = d
        import rjieba
        self.jieba = rjieba

    def get_vocab(self):
        if False:
            while True:
                i = 10
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, use_jieba=True):
        if False:
            i = 10
            return i + 15
        split_tokens = []
        if use_jieba:
            for wholword in self.jieba.cut(text, False):
                if wholword in self.vocab:
                    split_tokens.append(wholword)
                else:
                    char_list = self._tokenize(wholword, use_jieba=False)
                    split_tokens.extend(char_list)
        elif self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        if False:
            while True:
                i = 10
        'Converts a token (str) in an id using the vocab.'
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        if False:
            while True:
                i = 10
        'Converts an index (integer) in a token (str) using the vocab.'
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        if False:
            i = 10
            return i + 15
        'Converts a sequence of tokens (string) in a single string.'
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            return 10
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. A RoFormer sequence has the following format:\n\n        - single sequence: `[CLS] X [SEP]`\n        - pair of sequences: `[CLS] A [SEP] B [SEP]`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding\n        special tokens using the tokenizer `prepare_for_model` method.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n            already_has_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not the token list is already formatted with special tokens for the model.\n\n        Returns:\n            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.\n        '
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
        if token_ids_1 is not None:
            return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]
        return [1] + [0] * len(token_ids_0) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A RoFormer\n        sequence pair mask has the following format:\n\n        ```\n        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1\n        | first sequence    | second sequence |\n        ```\n\n        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            return 10
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        else:
            vocab_file = (filename_prefix + '-' if filename_prefix else '') + save_directory
        with open(vocab_file, 'w', encoding='utf-8') as writer:
            for (token, token_index) in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(f'Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive. Please check that the vocabulary is not corrupted!')
                    index = token_index
                writer.write(token + '\n')
                index += 1
        return (vocab_file,)