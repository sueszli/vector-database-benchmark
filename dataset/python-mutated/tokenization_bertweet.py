""" Tokenization classes for BERTweet"""
import html
import os
import re
from shutil import copyfile
from typing import List, Optional, Tuple
import regex
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt', 'merges_file': 'bpe.codes'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'vinai/bertweet-base': 'https://huggingface.co/vinai/bertweet-base/resolve/main/vocab.txt'}, 'merges_file': {'vinai/bertweet-base': 'https://huggingface.co/vinai/bertweet-base/resolve/main/bpe.codes'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'vinai/bertweet-base': 128}

def get_pairs(word):
    if False:
        i = 10
        return i + 15
    '\n    Return set of symbol pairs in a word.\n\n    Word is represented as tuple of symbols (symbols being variable-length strings).\n    '
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    pairs = set(pairs)
    return pairs

class BertweetTokenizer(PreTrainedTokenizer):
    """
    Constructs a BERTweet tokenizer, using Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        normalization (`bool`, *optional*, defaults to `False`):
            Whether or not to apply a normalization preprocess.
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
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, normalization=False, bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>', **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            from emoji import demojize
            self.demojizer = demojize
        except ImportError:
            logger.warning('emoji is not installed, thus not converting emoticons or emojis into text. Install emoji: pip3 install emoji==0.6.0')
            self.demojizer = None
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.encoder = {}
        self.encoder[str(bos_token)] = 0
        self.encoder[str(pad_token)] = 1
        self.encoder[str(eos_token)] = 2
        self.encoder[str(unk_token)] = 3
        self.add_from_file(vocab_file)
        self.decoder = {v: k for (k, v) in self.encoder.items()}
        with open(merges_file, encoding='utf-8') as merges_handle:
            merges = merges_handle.read().split('\n')[:-1]
        merges = [tuple(merge.split()[:-1]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        self.normalization = normalization
        self.tweetPreprocessor = TweetTokenizer()
        self.special_puncts = {'’': "'", '…': '...'}
        super().__init__(normalization=normalization, bos_token=bos_token, eos_token=eos_token, sep_token=sep_token, cls_token=cls_token, unk_token=unk_token, pad_token=pad_token, mask_token=mask_token, **kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            return 10
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. A BERTweet sequence has the following format:\n\n        - single sequence: `<s> X </s>`\n        - pair of sequences: `<s> A </s></s> B </s>`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            i = 10
            return i + 15
        '\n        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding\n        special tokens using the tokenizer `prepare_for_model` method.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n            already_has_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not the token list is already formatted with special tokens for the model.\n\n        Returns:\n            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.\n        '
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1, 1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BERTweet does\n        not make use of token type ids, therefore a list of zeros is returned.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of zeros.\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        if False:
            i = 10
            return i + 15
        return len(self.encoder)

    def get_vocab(self):
        if False:
            return 10
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if False:
            print('Hello World!')
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        word = tuple(list(word[:-1]) + [word[-1] + '</w>'])
        pairs = get_pairs(word)
        if not pairs:
            return token
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
        word = '@@ '.join(word)
        word = word[:-4]
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        if False:
            print('Hello World!')
        'Tokenize a string.'
        if self.normalization:
            text = self.normalizeTweet(text)
        split_tokens = []
        words = re.findall('\\S+\\n?', text)
        for token in words:
            split_tokens.extend(list(self.bpe(token).split(' ')))
        return split_tokens

    def normalizeTweet(self, tweet):
        if False:
            return 10
        '\n        Normalize a raw Tweet\n        '
        for punct in self.special_puncts:
            tweet = tweet.replace(punct, self.special_puncts[punct])
        tokens = self.tweetPreprocessor.tokenize(tweet)
        normTweet = ' '.join([self.normalizeToken(token) for token in tokens])
        normTweet = normTweet.replace('cannot ', 'can not ').replace("n't ", " n't ").replace("n 't ", " n't ").replace("ca n't", "can't").replace("ai n't", "ain't")
        normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ", " 'll ").replace("'d ", " 'd ").replace("'ve ", " 've ")
        normTweet = normTweet.replace(' p . m .', '  p.m.').replace(' p . m ', ' p.m ').replace(' a . m .', ' a.m.').replace(' a . m ', ' a.m ')
        return ' '.join(normTweet.split())

    def normalizeToken(self, token):
        if False:
            for i in range(10):
                print('nop')
        '\n        Normalize tokens in a Tweet\n        '
        lowercased_token = token.lower()
        if token.startswith('@'):
            return '@USER'
        elif lowercased_token.startswith('http') or lowercased_token.startswith('www'):
            return 'HTTPURL'
        elif len(token) == 1:
            if token in self.special_puncts:
                return self.special_puncts[token]
            if self.demojizer is not None:
                return self.demojizer(token)
            else:
                return token
        else:
            return token

    def _convert_token_to_id(self, token):
        if False:
            print('Hello World!')
        'Converts a token (str) in an id using the vocab.'
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        if False:
            while True:
                i = 10
        'Converts an index (integer) in a token (str) using the vocab.'
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        if False:
            for i in range(10):
                print('nop')
        'Converts a sequence of tokens (string) in a single string.'
        out_string = ' '.join(tokens).replace('@@ ', '').strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            i = 10
            return i + 15
        if not os.path.isdir(save_directory):
            logger.error(f'Vocabulary path ({save_directory}) should be a directory')
            return
        out_vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        out_merge_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['merges_file'])
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, 'wb') as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        if os.path.abspath(self.merges_file) != os.path.abspath(out_merge_file):
            copyfile(self.merges_file, out_merge_file)
        return (out_vocab_file, out_merge_file)

    def add_from_file(self, f):
        if False:
            for i in range(10):
                print('nop')
        '\n        Loads a pre-existing dictionary from a text file and adds its symbols to this instance.\n        '
        if isinstance(f, str):
            try:
                with open(f, 'r', encoding='utf-8') as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(f'Incorrect encoding detected in {f}, please rebuild the dataset')
            return
        lines = f.readlines()
        for lineTmp in lines:
            line = lineTmp.strip()
            idx = line.rfind(' ')
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            self.encoder[word] = len(self.encoder)
'\nTwitter-aware tokenizer, designed to be flexible and easy to adapt to new domains and tasks. The basic logic is this:\n\n1. The tuple regex_strings defines a list of regular expression strings.\n\n2. The regex_strings strings are put, in order, into a compiled regular expression object called word_re.\n\n3. The tokenization is done by word_re.findall(s), where s is the user-supplied string, inside the tokenize() method of\n   the class Tokenizer.\n\n4. When instantiating Tokenizer objects, there is a single option: preserve_case. By default, it is set to True. If it\n   is set to False, then the tokenizer will lowercase everything except for emoticons.\n\n'
EMOTICONS = "\n    (?:\n      [<>]?\n      [:;=8]                     # eyes\n      [\\-o\\*\\']?                 # optional nose\n      [\\)\\]\\(\\[dDpP/\\:\\}\\{@\\|\\\\] # mouth\n      |\n      [\\)\\]\\(\\[dDpP/\\:\\}\\{@\\|\\\\] # mouth\n      [\\-o\\*\\']?                 # optional nose\n      [:;=8]                     # eyes\n      [<>]?\n      |\n      <3                         # heart\n    )"
URLS = '\t\t\t# Capture 1: entire matched URL\n  (?:\n  https?:\t\t\t\t# URL protocol and colon\n    (?:\n      /{1,3}\t\t\t\t# 1-3 slashes\n      |\t\t\t\t\t#   or\n      [a-z0-9%]\t\t\t\t# Single letter or digit or \'%\'\n                                       # (Trying not to match e.g. "URI::Escape")\n    )\n    |\t\t\t\t\t#   or\n                                       # looks like domain name followed by a slash:\n    [a-z0-9.\\-]+[.]\n    (?:[a-z]{2,13})\n    /\n  )\n  (?:\t\t\t\t\t# One or more:\n    [^\\s()<>{}\\[\\]]+\t\t\t# Run of non-space, non-()<>{}[]\n    |\t\t\t\t\t#   or\n    \\([^\\s()]*?\\([^\\s()]+\\)[^\\s()]*?\\) # balanced parens, one level deep: (...(...)...)\n    |\n    \\([^\\s]+?\\)\t\t\t\t# balanced parens, non-recursive: (...)\n  )+\n  (?:\t\t\t\t\t# End with:\n    \\([^\\s()]*?\\([^\\s()]+\\)[^\\s()]*?\\) # balanced parens, one level deep: (...(...)...)\n    |\n    \\([^\\s]+?\\)\t\t\t\t# balanced parens, non-recursive: (...)\n    |\t\t\t\t\t#   or\n    [^\\s`!()\\[\\]{};:\'".,<>?«»“”‘’]\t# not a space or one of these punct chars\n  )\n  |\t\t\t\t\t# OR, the following to match naked domains:\n  (?:\n    (?<!@)\t\t\t        # not preceded by a @, avoid matching foo@_gmail.com_\n    [a-z0-9]+\n    (?:[.\\-][a-z0-9]+)*\n    [.]\n    (?:[a-z]{2,13})\n    \\b\n    /?\n    (?!@)\t\t\t        # not succeeded by a @,\n                            # avoid matching "foo.na" in "foo.na@example.com"\n  )\n'
REGEXPS = (URLS, '\n    (?:\n      (?:            # (international)\n        \\+?[01]\n        [ *\\-.\\)]*\n      )?\n      (?:            # (area code)\n        [\\(]?\n        \\d{3}\n        [ *\\-.\\)]*\n      )?\n      \\d{3}          # exchange\n      [ *\\-.\\)]*\n      \\d{4}          # base\n    )', EMOTICONS, '<[^>\\s]+>', '[\\-]+>|<[\\-]+', '(?:@[\\w_]+)', "(?:\\#+[\\w_]+[\\w\\'_\\-]*[\\w_]+)", '[\\w.+-]+@[\\w-]+\\.(?:[\\w-]\\.?)+[\\w-]', "\n    (?:[^\\W\\d_](?:[^\\W\\d_]|['\\-_])+[^\\W\\d_]) # Words with apostrophes or dashes.\n    |\n    (?:[+\\-]?\\d+[,/.:-]\\d+[+\\-]?)  # Numbers, including fractions, decimals.\n    |\n    (?:[\\w_]+)                     # Words without apostrophes or dashes.\n    |\n    (?:\\.(?:\\s*\\.){1,})            # Ellipsis dots.\n    |\n    (?:\\S)                         # Everything else that isn't whitespace.\n    ")
WORD_RE = regex.compile('(%s)' % '|'.join(REGEXPS), regex.VERBOSE | regex.I | regex.UNICODE)
HANG_RE = regex.compile('([^a-zA-Z0-9])\\1{3,}')
EMOTICON_RE = regex.compile(EMOTICONS, regex.VERBOSE | regex.I | regex.UNICODE)
ENT_RE = regex.compile('&(#?(x?))([^&;\\s]+);')

def _str_to_unicode(text, encoding=None, errors='strict'):
    if False:
        return 10
    if encoding is None:
        encoding = 'utf-8'
    if isinstance(text, bytes):
        return text.decode(encoding, errors)
    return text

def _replace_html_entities(text, keep=(), remove_illegal=True, encoding='utf-8'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove entities from text by converting them to their corresponding unicode character.\n\n    Args:\n        text:\n            A unicode string or a byte string encoded in the given *encoding* (which defaults to \'utf-8\').\n        keep (list):\n            List of entity names which should not be replaced. This supports both numeric entities (`&#nnnn;` and\n            `&#hhhh;`) and named entities (such as `&nbsp;` or `&gt;`).\n        remove_illegal (bool):\n            If `True`, entities that can\'t be converted are removed. Otherwise, entities that can\'t be converted are\n            kept "as is".\n\n    Returns: A unicode string with the entities removed.\n\n    See https://github.com/scrapy/w3lib/blob/master/w3lib/html.py\n\n    Examples:\n\n    ```python\n    >>> from nltk.tokenize.casual import _replace_html_entities\n\n    >>> _replace_html_entities(b"Price: &pound;100")\n    \'Price: \\xa3100\'\n\n    >>> print(_replace_html_entities(b"Price: &pound;100"))\n    Price: £100\n    ```'

    def _convert_entity(match):
        if False:
            i = 10
            return i + 15
        entity_body = match.group(3)
        if match.group(1):
            try:
                if match.group(2):
                    number = int(entity_body, 16)
                else:
                    number = int(entity_body, 10)
                if 128 <= number <= 159:
                    return bytes((number,)).decode('cp1252')
            except ValueError:
                number = None
        elif entity_body in keep:
            return match.group(0)
        else:
            number = html.entities.name2codepoint.get(entity_body)
        if number is not None:
            try:
                return chr(number)
            except (ValueError, OverflowError):
                pass
        return '' if remove_illegal else match.group(0)
    return ENT_RE.sub(_convert_entity, _str_to_unicode(text, encoding))

class TweetTokenizer:
    """
    Examples:

    ```python
    >>> # Tokenizer for tweets.
    >>> from nltk.tokenize import TweetTokenizer

    >>> tknzr = TweetTokenizer()
    >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
    >>> tknzr.tokenize(s0)
    ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->', '<--']

    >>> # Examples using *strip_handles* and *reduce_len parameters*:
    >>> tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    >>> s1 = "@remy: This is waaaaayyyy too much for you!!!!!!"
    >>> tknzr.tokenize(s1)
    [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']
    ```"""

    def __init__(self, preserve_case=True, reduce_len=False, strip_handles=False):
        if False:
            return 10
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles

    def tokenize(self, text):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            text: str\n\n        Returns: list(str) A tokenized list of strings; concatenating this list returns the original string if\n        `preserve_case=False`\n        '
        text = _replace_html_entities(text)
        if self.strip_handles:
            text = remove_handles(text)
        if self.reduce_len:
            text = reduce_lengthening(text)
        safe_text = HANG_RE.sub('\\1\\1\\1', text)
        words = WORD_RE.findall(safe_text)
        if not self.preserve_case:
            words = [x if EMOTICON_RE.search(x) else x.lower() for x in words]
        return words

def reduce_lengthening(text):
    if False:
        return 10
    '\n    Replace repeated character sequences of length 3 or greater with sequences of length 3.\n    '
    pattern = regex.compile('(.)\\1{2,}')
    return pattern.sub('\\1\\1\\1', text)

def remove_handles(text):
    if False:
        return 10
    '\n    Remove Twitter username handles from text.\n    '
    pattern = regex.compile('(?<![A-Za-z0-9_!@#\\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)')
    return pattern.sub(' ', text)

def casual_tokenize(text, preserve_case=True, reduce_len=False, strip_handles=False):
    if False:
        while True:
            i = 10
    '\n    Convenience function for wrapping the tokenizer.\n    '
    return TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len, strip_handles=strip_handles).tokenize(text)