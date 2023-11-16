"""Tokenization classes for Flaubert."""
import json
import os
import re
import unicodedata
from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'merges_file': 'merges.txt'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'flaubert/flaubert_small_cased': 'https://huggingface.co/flaubert/flaubert_small_cased/resolve/main/vocab.json', 'flaubert/flaubert_base_uncased': 'https://huggingface.co/flaubert/flaubert_base_uncased/resolve/main/vocab.json', 'flaubert/flaubert_base_cased': 'https://huggingface.co/flaubert/flaubert_base_cased/resolve/main/vocab.json', 'flaubert/flaubert_large_cased': 'https://huggingface.co/flaubert/flaubert_large_cased/resolve/main/vocab.json'}, 'merges_file': {'flaubert/flaubert_small_cased': 'https://huggingface.co/flaubert/flaubert_small_cased/resolve/main/merges.txt', 'flaubert/flaubert_base_uncased': 'https://huggingface.co/flaubert/flaubert_base_uncased/resolve/main/merges.txt', 'flaubert/flaubert_base_cased': 'https://huggingface.co/flaubert/flaubert_base_cased/resolve/main/merges.txt', 'flaubert/flaubert_large_cased': 'https://huggingface.co/flaubert/flaubert_large_cased/resolve/main/merges.txt'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'flaubert/flaubert_small_cased': 512, 'flaubert/flaubert_base_uncased': 512, 'flaubert/flaubert_base_cased': 512, 'flaubert/flaubert_large_cased': 512}
PRETRAINED_INIT_CONFIGURATION = {'flaubert/flaubert_small_cased': {'do_lowercase': False}, 'flaubert/flaubert_base_uncased': {'do_lowercase': True}, 'flaubert/flaubert_base_cased': {'do_lowercase': False}, 'flaubert/flaubert_large_cased': {'do_lowercase': False}}

def convert_to_unicode(text):
    if False:
        i = 10
        return i + 15
    "\n    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.\n    "

    def ensure_text(s, encoding='utf-8', errors='strict'):
        if False:
            while True:
                i = 10
        if isinstance(s, bytes):
            return s.decode(encoding, errors)
        elif isinstance(s, str):
            return s
        else:
            raise TypeError(f"not expecting type '{type(s)}'")
    return ensure_text(text, encoding='utf-8', errors='ignore')

def get_pairs(word):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length\n    strings)\n    '
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def replace_unicode_punct(text):
    if False:
        while True:
            i = 10
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

class FlaubertTokenizer(PreTrainedTokenizer):
    """
    Construct a Flaubert tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization.
    - Normalizing all inputs text.
    - The arguments `special_tokens` and the function `set_special_tokens`, can be used to add additional symbols (like
      "__classify__") to a vocabulary.
    - The argument `do_lowercase` controls lower casing (automatically set for pretrained vocabularies).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Vocabulary file.
        merges_file (`str`):
            Merges file.
        do_lowercase (`bool`, *optional*, defaults to `False`):
            Controls lower casing.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"</s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<special1>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (`List[str]`, *optional*, defaults to `['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>']`):
            List of additional special tokens.
        lang2id (`Dict[str, int]`, *optional*):
            Dictionary mapping languages string identifiers to their IDs.
        id2lang (`Dict[int, str]`, *optional*):
            Dictionary mapping language IDs to their string identifiers.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, do_lowercase=False, unk_token='<unk>', bos_token='<s>', sep_token='</s>', pad_token='<pad>', cls_token='</s>', mask_token='<special1>', additional_special_tokens=['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>'], lang2id=None, id2lang=None, **kwargs):
        if False:
            while True:
                i = 10
        do_lowercase_and_remove_accent = kwargs.pop('do_lowercase_and_remove_accent', None)
        if do_lowercase_and_remove_accent is not None:
            logger.warning("`do_lowercase_and_remove_accent` is passed as a keyword argument, but this won't do anything. `FlaubertTokenizer` will always set it to `False`.")
        self.do_lowercase_and_remove_accent = False
        self.do_lowercase = do_lowercase
        try:
            import sacremoses
        except ImportError:
            raise ImportError('You need to install sacremoses to use FlaubertTokenizer. See https://pypi.org/project/sacremoses/ for installation.')
        self.sm = sacremoses
        self.cache_moses_punct_normalizer = {}
        self.cache_moses_tokenizer = {}
        self.lang_with_custom_tokenizer = {'zh', 'th', 'ja'}
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
        super().__init__(unk_token=unk_token, bos_token=bos_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, additional_special_tokens=additional_special_tokens, lang2id=lang2id, id2lang=id2lang, **kwargs)

    @property
    def do_lower_case(self):
        if False:
            i = 10
            return i + 15
        return self.do_lowercase_and_remove_accent

    def moses_punct_norm(self, text, lang):
        if False:
            return 10
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = self.sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        return punct_normalizer.normalize(text)

    def moses_tokenize(self, text, lang):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
        return len(self.encoder)

    def get_vocab(self):
        if False:
            while True:
                i = 10
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if False:
            for i in range(10):
                print('nop')
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

    def preprocess_text(self, text):
        if False:
            print('Hello World!')
        text = text.replace('``', '"').replace("''", '"')
        text = convert_to_unicode(text)
        text = unicodedata.normalize('NFC', text)
        if self.do_lowercase:
            text = text.lower()
        return text

    def _tokenize(self, text, bypass_tokenizer=False):
        if False:
            print('Hello World!')
        '\n        Tokenize a string given language code using Moses.\n\n        Details of tokenization:\n\n            - [sacremoses](https://github.com/alvations/sacremoses): port of Moses\n            - Install with `pip install sacremoses`\n\n        Args:\n            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)\n              (bool). If True, we only apply BPE.\n\n        Returns:\n            List of tokens.\n        '
        lang = 'fr'
        if lang and self.lang2id and (lang not in self.lang2id):
            logger.error('Supplied language code not found in lang2id mapping. Please check that your language is supported by the loaded pretrained model.')
        if bypass_tokenizer:
            text = text.split()
        else:
            text = self.preprocess_text(text)
            text = self.moses_pipeline(text, lang=lang)
            text = self.moses_tokenize(text, lang=lang)
        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend(list(self.bpe(token).split(' ')))
        return split_tokens

    def _convert_token_to_id(self, token):
        if False:
            i = 10
            return i + 15
        'Converts a token (str) in an id using the vocab.'
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        if False:
            return 10
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
            print('Hello World!')
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. An XLM sequence has the following format:\n\n        - single sequence: `<s> X </s>`\n        - pair of sequences: `<s> A </s> B </s>`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n\n        '
        bos = [self.bos_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return bos + token_ids_0 + sep
        return bos + token_ids_0 + sep + token_ids_1 + sep

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
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
        state = self.__dict__.copy()
        state['sm'] = None
        return state

    def __setstate__(self, d):
        if False:
            return 10
        self.__dict__ = d
        try:
            import sacremoses
        except ImportError:
            raise ImportError('You need to install sacremoses to use XLMTokenizer. See https://pypi.org/project/sacremoses/ for installation.')
        self.sm = sacremoses