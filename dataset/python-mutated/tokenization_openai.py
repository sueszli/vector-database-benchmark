"""Tokenization classes for OpenAI GPT."""
import json
import os
import re
import unicodedata
from typing import Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'merges_file': 'merges.txt'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'openai-gpt': 'https://huggingface.co/openai-gpt/resolve/main/vocab.json'}, 'merges_file': {'openai-gpt': 'https://huggingface.co/openai-gpt/resolve/main/merges.txt'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'openai-gpt': 512}

def whitespace_tokenize(text):
    if False:
        print('Hello World!')
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
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
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
            return 10
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
            while True:
                i = 10
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
            print('Hello World!')
        'Checks whether CP is the codepoint of a CJK character.'
        if cp >= 19968 and cp <= 40959 or (cp >= 13312 and cp <= 19903) or (cp >= 131072 and cp <= 173791) or (cp >= 173824 and cp <= 177983) or (cp >= 177984 and cp <= 178207) or (cp >= 178208 and cp <= 183983) or (cp >= 63744 and cp <= 64255) or (cp >= 194560 and cp <= 195103):
            return True
        return False

    def _clean_text(self, text):
        if False:
            i = 10
            return i + 15
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

def get_pairs(word):
    if False:
        return 10
    '\n    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length\n    strings)\n    '
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def text_standardize(text):
    if False:
        for i in range(10):
            print('nop')
    '\n    fixes some issues the spacy tokenizer had on books corpus also does some whitespace standardization\n    '
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub('(-+|~+|!+|"+|;+|\\?+|\\++|,+|\\)+|\\(+|\\\\+|\\/+|\\*+|\\[+|\\]+|}+|{+|\\|+|_+)', ' \\1 ', text)
    text = re.sub('\\s*\\n\\s*', ' \n ', text)
    text = re.sub('[^\\S\\n]+', ' ', text)
    return text.strip()

class OpenAIGPTTokenizer(PreTrainedTokenizer):
    """
    Construct a GPT Tokenizer. Based on Byte-Pair-Encoding with the following peculiarities:

    - lowercases all inputs,
    - uses `SpaCy` tokenizer and `ftfy` for pre-BPE tokenization if they are installed, fallback to BERT's
      `BasicTokenizer` if not.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, vocab_file, merges_file, unk_token='<unk>', **kwargs):
        if False:
            return 10
        try:
            import ftfy
            from spacy.lang.en import English
            _nlp = English()
            self.nlp = _nlp.tokenizer
            self.fix_text = ftfy.fix_text
        except ImportError:
            logger.warning('ftfy or spacy is not installed using BERT BasicTokenizer instead of SpaCy & ftfy.')
            self.nlp = BasicTokenizer(do_lower_case=True)
            self.fix_text = None
        with open(vocab_file, encoding='utf-8') as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for (k, v) in self.encoder.items()}
        with open(merges_file, encoding='utf-8') as merges_handle:
            merges = merges_handle.read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        super().__init__(unk_token=unk_token, **kwargs)

    @property
    def do_lower_case(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    @property
    def vocab_size(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.encoder)

    def get_vocab(self):
        if False:
            return 10
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if False:
            print('Hello World!')
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)
        if not pairs:
            return token + '</w>'
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            (first, second) = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                if word[i] == first and i < len(word) - 1 and (word[i + 1] == second):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        if False:
            print('Hello World!')
        'Tokenize a string.'
        split_tokens = []
        if self.fix_text is None:
            text = self.nlp.tokenize(text)
            for token in text:
                split_tokens.extend(list(self.bpe(token).split(' ')))
        else:
            text = self.nlp(text_standardize(self.fix_text(text)))
            for token in text:
                split_tokens.extend(list(self.bpe(token.text.lower()).split(' ')))
        return split_tokens

    def _convert_token_to_id(self, token):
        if False:
            for i in range(10):
                print('nop')
        'Converts a token (str) in an id using the vocab.'
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        if False:
            while True:
                i = 10
        'Converts an id in a token (BPE) using the vocab.'
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        if False:
            i = 10
            return i + 15
        'Converts a sequence of tokens (string) in a single string.'
        out_string = ''.join(tokens).replace('</w>', ' ').strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            return 10
        if not os.path.isdir(save_directory):
            logger.error(f'Vocabulary path ({save_directory}) should be a directory')
            return
        vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        merge_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['merges_file'])
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
        index = 0
        with open(merge_file, 'w', encoding='utf-8') as writer:
            writer.write('#version: 0.2\n')
            for (bpe_tokens, token_index) in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(f'Saving vocabulary to {merge_file}: BPE merge indices are not consecutive. Please check that the tokenizer is not corrupted!')
                    index = token_index
                writer.write(' '.join(bpe_tokens) + '\n')
                index += 1
        return (vocab_file, merge_file)