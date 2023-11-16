"""Tokenization classes. Provided as is from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py"""
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import logging
import os
import unicodedata
from io import open
from .file_utils import cached_path
logger = logging.getLogger(__name__)
PRETRAINED_VOCAB_ARCHIVE_MAP = {'bert-base-uncased': '.pytorch_pretrained_bert/bert-base-uncased-vocab.txt', 'bert-large-uncased': '.pytorch_pretrained_bert/bert-large-uncased-vocab.txt', 'bert-base-cased': '.pytorch_pretrained_bert/bert-base-cased-vocab.txt', 'bert-large-cased': '.pytorch_pretrained_bert/bert-large-cased-vocab.txt', 'bert-base-multilingual-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt', 'bert-base-multilingual-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt', 'bert-base-chinese': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt'}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {'bert-base-uncased': 512, 'bert-large-uncased': 512, 'bert-base-cased': 512, 'bert-large-cased': 512, 'bert-base-multilingual-uncased': 512, 'bert-base-multilingual-cased': 512, 'bert-base-chinese': 512}
VOCAB_NAME = 'vocab.txt'

def load_vocab(vocab_file):
    if False:
        while True:
            i = 10
    'Loads a vocabulary file into a dictionary.'
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def whitespace_tokenize(text):
    if False:
        print('Hello World!')
    'Runs basic whitespace cleaning and splitting on a piece of text.'
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True, never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')):
        if False:
            i = 10
            return i + 15
        "Constructs a BertTokenizer.\n\n        Args:\n          vocab_file: Path to a one-wordpiece-per-line vocabulary file\n          do_lower_case: Whether to lower case the input\n                         Only has an effect when do_wordpiece_only=False\n          do_basic_tokenize: Whether to do basic tokenization before wordpiece.\n          max_len: An artificial maximum length to truncate tokenized sequences to;\n                         Effective maximum length is always the minimum of this\n                         value (if specified) and the underlying BERT model's\n                         sequence length.\n          never_split: List of tokens which will never be split during tokenization.\n                         Only has an effect when do_wordpiece_only=False\n        "
        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for (tok, ids) in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1000000000000.0)

    def tokenize(self, text):
        if False:
            while True:
                i = 10
        if self.do_basic_tokenize:
            split_tokens = []
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        if False:
            while True:
                i = 10
        'Converts a sequence of tokens into ids using the vocab.'
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            logger.warning('Token indices sequence length is longer than the specified maximum  sequence length for this BERT model ({} > {}). Running this sequence through BERT will result in indexing errors'.format(len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids):
        if False:
            i = 10
            return i + 15
        'Converts a sequence of ids in wordpiece tokens using the vocab.'
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        if False:
            print('Hello World!')
        '\n        Instantiate a PreTrainedBertModel from a pre-trained model file.\n        Download and cache the pre-trained model file if needed.\n        '
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            vocab_file = pretrained_model_name_or_path
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error("Model name '{}' was not found in model name list ({}). We assumed '{}' was a path or url but couldn't find any file associated to this path or url.".format(pretrained_model_name_or_path, ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()), vocab_file))
            return None
        if resolved_vocab_file == vocab_file:
            logger.info('loading vocabulary file {}'.format(vocab_file))
        else:
            logger.info('loading vocabulary file {} from cache at {}'.format(vocab_file, resolved_vocab_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1000000000000.0)), max_len)
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
        return tokenizer

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')):
        if False:
            i = 10
            return i + 15
        'Constructs a BasicTokenizer.\n\n        Args:\n          do_lower_case: Whether to lower case the input.\n        '
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        if False:
            while True:
                i = 10
        'Tokenizes a piece of text.'
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
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

    def _run_split_on_punc(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Splits punctuation on a piece of text.'
        if text in self.never_split:
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
            print('Hello World!')
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
            return 10
        'Checks whether CP is the codepoint of a CJK character.'
        if cp >= 19968 and cp <= 40959 or (cp >= 13312 and cp <= 19903) or (cp >= 131072 and cp <= 173791) or (cp >= 173824 and cp <= 177983) or (cp >= 177984 and cp <= 178207) or (cp >= 178208 and cp <= 183983) or (cp >= 63744 and cp <= 64255) or (cp >= 194560 and cp <= 195103):
            return True
        return False

    def _clean_text(self, text):
        if False:
            return 10
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

    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=100):
        if False:
            while True:
                i = 10
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        if False:
            return 10
        'Tokenizes a piece of text into its word pieces.\n\n        This uses a greedy longest-match-first algorithm to perform tokenization\n        using the given vocabulary.\n\n        For example:\n          >>> input = "unaffable"\n          >>> output = ["un", "##aff", "##able"]\n\n        Args:\n          text: A single token or whitespace separated tokens. This should have\n            already been passed through `BasicTokenizer`.\n\n        Returns:\n          A list of wordpiece tokens.\n        '
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

def _is_whitespace(char):
    if False:
        i = 10
        return i + 15
    'Checks whether `chars` is a whitespace character.'
    if char == ' ' or char == '\t' or char == '\n' or (char == '\r'):
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False

def _is_control(char):
    if False:
        print('Hello World!')
    'Checks whether `chars` is a control character.'
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False

def _is_punctuation(char):
    if False:
        print('Hello World!')
    'Checks whether `chars` is a punctuation character.'
    cp = ord(char)
    if cp >= 33 and cp <= 47 or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False