import json
import os
import re
import unicodedata
from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'merges_file': 'merges.txt'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'allegro/herbert-base-cased': 'https://huggingface.co/allegro/herbert-base-cased/resolve/main/vocab.json'}, 'merges_file': {'allegro/herbert-base-cased': 'https://huggingface.co/allegro/herbert-base-cased/resolve/main/merges.txt'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'allegro/herbert-base-cased': 514}
PRETRAINED_INIT_CONFIGURATION = {}

def get_pairs(word):
    if False:
        while True:
            i = 10
    '\n    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length\n    strings)\n    '
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def replace_unicode_punct(text):
    if False:
        i = 10
        return i + 15
    '\n    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl\n    '
    text = text.replace('，', ',')
    text = re.sub('。\\s*', '. ', text)
    text = text.replace('、', ',')
    text = text.replace('”', '"')
    text = text.replace('“', '"')
    text = text.replace('∶', ':')
    text = text.replace('：', ':')
    text = text.replace('？', '?')
    text = text.replace('《', '"')
    text = text.replace('》', '"')
    text = text.replace('）', ')')
    text = text.replace('！', '!')
    text = text.replace('（', '(')
    text = text.replace('；', ';')
    text = text.replace('１', '1')
    text = text.replace('」', '"')
    text = text.replace('「', '"')
    text = text.replace('０', '0')
    text = text.replace('３', '3')
    text = text.replace('２', '2')
    text = text.replace('５', '5')
    text = text.replace('６', '6')
    text = text.replace('９', '9')
    text = text.replace('７', '7')
    text = text.replace('８', '8')
    text = text.replace('４', '4')
    text = re.sub('．\\s*', '. ', text)
    text = text.replace('～', '~')
    text = text.replace('’', "'")
    text = text.replace('…', '...')
    text = text.replace('━', '-')
    text = text.replace('〈', '<')
    text = text.replace('〉', '>')
    text = text.replace('【', '[')
    text = text.replace('】', ']')
    text = text.replace('％', '%')
    return text

def remove_non_printing_char(text):
    if False:
        return 10
    '\n    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl\n    '
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith('C'):
            continue
        output.append(char)
    return ''.join(output)

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
            return 10
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
            i = 10
            return i + 15
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
            i = 10
            return i + 15
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
            return 10
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
            i = 10
            return i + 15
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

class HerbertTokenizer(PreTrainedTokenizer):
    """
    Construct a BPE tokenizer for HerBERT.

    Peculiarities:

    - uses BERT's pre-tokenizer: BaseTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of a
      punctuation character will be treated separately.

    - Such pretokenized input is BPE subtokenized

    This tokenizer inherits from [`XLMTokenizer`] which contains most of the methods. Users should refer to the
    superclass for more information regarding methods.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, tokenizer_file=None, cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>', sep_token='</s>', bos_token='<s>', do_lowercase_and_remove_accent=False, additional_special_tokens=['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>'], lang2id=None, id2lang=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        try:
            import sacremoses
        except ImportError:
            raise ImportError('You need to install sacremoses to use HerbertTokenizer. See https://pypi.org/project/sacremoses/ for installation.')
        self.sm = sacremoses
        self.cache_moses_punct_normalizer = {}
        self.cache_moses_tokenizer = {}
        self.lang_with_custom_tokenizer = {'zh', 'th', 'ja'}
        self.do_lowercase_and_remove_accent = do_lowercase_and_remove_accent
        self.lang2id = lang2id
        self.id2lang = id2lang
        if lang2id is not None and id2lang is not None:
            assert len(lang2id) == len(id2lang)
        self.ja_word_tokenizer = None
        self.zh_word_tokenizer = None
        with open(vocab_file, encoding='utf-8') as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for (k, v) in self.encoder.items()}
        with open(merges_file, encoding='utf-8') as merges_handle:
            merges = merges_handle.read().split('\n')[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        super().__init__(unk_token=unk_token, bos_token=bos_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, additional_special_tokens=additional_special_tokens, lang2id=lang2id, id2lang=id2lang, do_lowercase_and_remove_accent=do_lowercase_and_remove_accent, tokenizer_file=None, **kwargs)
        self.bert_pre_tokenizer = BasicTokenizer(do_lower_case=False, never_split=self.all_special_tokens, tokenize_chinese_chars=False, strip_accents=False)

    @property
    def do_lower_case(self):
        if False:
            while True:
                i = 10
        return self.do_lowercase_and_remove_accent

    def moses_punct_norm(self, text, lang):
        if False:
            i = 10
            return i + 15
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = self.sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        return punct_normalizer.normalize(text)

    def moses_tokenize(self, text, lang):
        if False:
            i = 10
            return i + 15
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)

    def moses_pipeline(self, text, lang):
        if False:
            return 10
        text = replace_unicode_punct(text)
        text = self.moses_punct_norm(text, lang)
        text = remove_non_printing_char(text)
        return text

    def ja_tokenize(self, text):
        if False:
            for i in range(10):
                print('nop')
        if self.ja_word_tokenizer is None:
            try:
                import Mykytea
                self.ja_word_tokenizer = Mykytea.Mykytea(f"-model {os.path.expanduser('~')}/local/share/kytea/model.bin")
            except (AttributeError, ImportError):
                logger.error("Make sure you install KyTea (https://github.com/neubig/kytea) and it's python wrapper (https://github.com/chezou/Mykytea-python) with the following steps")
                logger.error('1. git clone git@github.com:neubig/kytea.git && cd kytea')
                logger.error('2. autoreconf -i')
                logger.error('3. ./configure --prefix=$HOME/local')
                logger.error('4. make && make install')
                logger.error('5. pip install kytea')
                raise
        return list(self.ja_word_tokenizer.getWS(text))

    @property
    def vocab_size(self):
        if False:
            return 10
        return len(self.encoder)

    def get_vocab(self):
        if False:
            while True:
                i = 10
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if False:
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        pre_tokens = self.bert_pre_tokenizer.tokenize(text)
        split_tokens = []
        for token in pre_tokens:
            if token:
                split_tokens.extend(list(self.bpe(token).split(' ')))
        return split_tokens

    def _convert_token_to_id(self, token):
        if False:
            return 10
        'Converts a token (str) in an id using the vocab.'
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        if False:
            print('Hello World!')
        'Converts an index (integer) in a token (str) using the vocab.'
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        if False:
            i = 10
            return i + 15
        'Converts a sequence of tokens (string) in a single string.'
        out_string = ''.join(tokens).replace('</w>', ' ').strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            return 10
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. An XLM sequence has the following format:\n\n        - single sequence: `<s> X </s>`\n        - pair of sequences: `<s> A </s> B </s>`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n\n        '
        bos = [self.bos_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return bos + token_ids_0 + sep
        return bos + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            while True:
                i = 10
        '\n        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding\n        special tokens using the tokenizer `prepare_for_model` method.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n            already_has_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not the token list is already formatted with special tokens for the model.\n\n        Returns:\n            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.\n        '
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
        if token_ids_1 is not None:
            return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]
        return [1] + [0] * len(token_ids_0) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLM sequence\n        pair mask has the following format:\n\n        ```\n        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1\n        | first sequence    | second sequence |\n        ```\n\n        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            for i in range(10):
                print('nop')
        if not os.path.isdir(save_directory):
            logger.error(f'Vocabulary path ({save_directory}) should be a directory')
            return
        vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        merge_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['merges_file'])
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
        index = 0
        with open(merge_file, 'w', encoding='utf-8') as writer:
            for (bpe_tokens, token_index) in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(f'Saving vocabulary to {merge_file}: BPE merge indices are not consecutive. Please check that the tokenizer is not corrupted!')
                    index = token_index
                writer.write(' '.join(bpe_tokens) + '\n')
                index += 1
        return (vocab_file, merge_file)

    def __getstate__(self):
        if False:
            return 10
        state = self.__dict__.copy()
        state['sm'] = None
        return state

    def __setstate__(self, d):
        if False:
            while True:
                i = 10
        self.__dict__ = d
        try:
            import sacremoses
        except ImportError:
            raise ImportError('You need to install sacremoses to use XLMTokenizer. See https://pypi.org/project/sacremoses/ for installation.')
        self.sm = sacremoses