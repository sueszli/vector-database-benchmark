"""Tokenization class for MarkupLM."""
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import regex as re
from ...file_utils import PaddingStrategy, TensorType, add_end_docstrings
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import ENCODE_KWARGS_DOCSTRING, BatchEncoding, EncodedInput, PreTokenizedInput, TextInput, TextInputPair, TruncationStrategy
from ...utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'merges_file': 'merges.txt', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'microsoft/markuplm-base': 'https://huggingface.co/microsoft/markuplm-base/resolve/main/vocab.json', 'microsoft/markuplm-large': 'https://huggingface.co/microsoft/markuplm-large/resolve/main/vocab.json'}, 'merges_file': {'microsoft/markuplm-base': 'https://huggingface.co/microsoft/markuplm-base/resolve/main/merges.txt', 'microsoft/markuplm-large': 'https://huggingface.co/microsoft/markuplm-large/resolve/main/merges.txt'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'microsoft/markuplm-base': 512, 'microsoft/markuplm-large': 512}
MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = "\n            add_special_tokens (`bool`, *optional*, defaults to `True`):\n                Whether or not to encode the sequences with the special tokens relative to their model.\n            padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `False`):\n                Activates and controls padding. Accepts the following values:\n\n                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single\n                  sequence if provided).\n                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum\n                  acceptable input length for the model if that argument is not provided.\n                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different\n                  lengths).\n            truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):\n                Activates and controls truncation. Accepts the following values:\n\n                - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or\n                  to the maximum acceptable input length for the model if that argument is not provided. This will\n                  truncate token by token, removing a token from the longest sequence in the pair if a pair of\n                  sequences (or a batch of pairs) is provided.\n                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the\n                  maximum acceptable input length for the model if that argument is not provided. This will only\n                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.\n                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the\n                  maximum acceptable input length for the model if that argument is not provided. This will only\n                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.\n                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths\n                  greater than the model maximum admissible input size).\n            max_length (`int`, *optional*):\n                Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to\n                `None`, this will use the predefined model maximum length if a maximum length is required by one of the\n                truncation/padding parameters. If the model has no specific maximum input length (like XLNet)\n                truncation/padding to a maximum length will be deactivated.\n            stride (`int`, *optional*, defaults to 0):\n                If set to a number along with `max_length`, the overflowing tokens returned when\n                `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence\n                returned to provide some overlap between truncated and overflowing sequences. The value of this\n                argument defines the number of overlapping tokens.\n            pad_to_multiple_of (`int`, *optional*):\n                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable\n                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).\n            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):\n                If set, will return tensors instead of list of python integers. Acceptable values are:\n\n                - `'tf'`: Return TensorFlow `tf.constant` objects.\n                - `'pt'`: Return PyTorch `torch.Tensor` objects.\n                - `'np'`: Return Numpy `np.ndarray` objects.\n"

@lru_cache()
def bytes_to_unicode():
    if False:
        print('Hello World!')
    "\n    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control\n    characters the bpe code barfs on. The reversible bpe codes work on unicode strings. This means you need a large #\n    of unicode characters in your vocab if you want to avoid UNKs. When you're at something like a 10B token dataset\n    you end up needing around 5K for decent coverage. This is a significant percentage of your normal, say, 32K bpe\n    vocab. To avoid that, we want lookup tables between utf-8 bytes and unicode strings.\n    "
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return set of symbol pairs in a word. Word is represented as tuple of symbols (symbols being variable-length\n    strings).\n    '
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class MarkupLMTokenizer(PreTrainedTokenizer):
    """
    Construct a MarkupLM tokenizer. Based on byte-level Byte-Pair-Encoding (BPE). [`MarkupLMTokenizer`] can be used to
    turn HTML strings into to token-level `input_ids`, `attention_mask`, `token_type_ids`, `xpath_tags_seq` and
    `xpath_tags_seq`. This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
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
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, tags_dict, errors='replace', bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>', add_prefix_space=False, max_depth=50, max_width=1000, pad_width=1001, pad_token_label=-100, only_label_first_subword=True, **kwargs):
        if False:
            return 10
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        with open(vocab_file, encoding='utf-8') as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.tags_dict = tags_dict
        self.decoder = {v: k for (k, v) in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for (k, v) in self.byte_encoder.items()}
        with open(merges_file, encoding='utf-8') as merges_handle:
            bpe_merges = merges_handle.read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space
        self.pat = re.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")
        self.max_depth = max_depth
        self.max_width = max_width
        self.pad_width = pad_width
        self.unk_tag_id = len(self.tags_dict)
        self.pad_tag_id = self.unk_tag_id + 1
        self.pad_xpath_tags_seq = [self.pad_tag_id] * self.max_depth
        self.pad_xpath_subs_seq = [self.pad_width] * self.max_depth
        super().__init__(vocab_file=vocab_file, merges_file=merges_file, tags_dict=tags_dict, errors=errors, bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, sep_token=sep_token, cls_token=cls_token, pad_token=pad_token, mask_token=mask_token, add_prefix_space=add_prefix_space, max_depth=max_depth, max_width=max_width, pad_width=pad_width, pad_token_label=pad_token_label, only_label_first_subword=only_label_first_subword, **kwargs)
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

    def get_xpath_seq(self, xpath):
        if False:
            while True:
                i = 10
        '\n        Given the xpath expression of one particular node (like "/html/body/div/li[1]/div/span[2]"), return a list of\n        tag IDs and corresponding subscripts, taking into account max depth.\n        '
        xpath_tags_list = []
        xpath_subs_list = []
        xpath_units = xpath.split('/')
        for unit in xpath_units:
            if not unit.strip():
                continue
            name_subs = unit.strip().split('[')
            tag_name = name_subs[0]
            sub = 0 if len(name_subs) == 1 else int(name_subs[1][:-1])
            xpath_tags_list.append(self.tags_dict.get(tag_name, self.unk_tag_id))
            xpath_subs_list.append(min(self.max_width, sub))
        xpath_tags_list = xpath_tags_list[:self.max_depth]
        xpath_subs_list = xpath_subs_list[:self.max_depth]
        xpath_tags_list += [self.pad_tag_id] * (self.max_depth - len(xpath_tags_list))
        xpath_subs_list += [self.pad_width] * (self.max_depth - len(xpath_subs_list))
        return (xpath_tags_list, xpath_subs_list)

    @property
    def vocab_size(self):
        if False:
            print('Hello World!')
        return len(self.encoder)

    def get_vocab(self):
        if False:
            while True:
                i = 10
        vocab = self.encoder.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def bpe(self, token):
        if False:
            i = 10
            return i + 15
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
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
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        if False:
            return 10
        'Tokenize a string.'
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join((self.byte_encoder[b] for b in token.encode('utf-8')))
            bpe_tokens.extend((bpe_token for bpe_token in self.bpe(token).split(' ')))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        if False:
            for i in range(10):
                print('nop')
        'Converts a token (str) in an id using the vocab.'
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        if False:
            return 10
        'Converts an index (integer) in a token (str) using the vocab.'
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        if False:
            for i in range(10):
                print('nop')
        'Converts a sequence of tokens (string) in a single string.'
        logger.warning('MarkupLM now does not support generative tasks, decoding is experimental and subject to change.')
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

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
            writer.write('#version: 0.2\n')
            for (bpe_tokens, token_index) in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(f'Saving vocabulary to {merge_file}: BPE merge indices are not consecutive. Please check that the tokenizer is not corrupted!')
                    index = token_index
                writer.write(' '.join(bpe_tokens) + '\n')
                index += 1
        return (vocab_file, merge_file)

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        if False:
            return 10
        add_prefix_space = kwargs.pop('add_prefix_space', self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and (not text[0].isspace())):
            text = ' ' + text
        return (text, kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            return 10
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. A RoBERTa sequence has the following format:\n        - single sequence: `<s> X </s>`\n        - pair of sequences: `<s> A </s></s> B </s>`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def build_xpath_tags_with_special_tokens(self, xpath_tags_0: List[int], xpath_tags_1: Optional[List[int]]=None) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        pad = [self.pad_xpath_tags_seq]
        if len(xpath_tags_1) == 0:
            return pad + xpath_tags_0 + pad
        return pad + xpath_tags_0 + pad + xpath_tags_1 + pad

    def build_xpath_subs_with_special_tokens(self, xpath_subs_0: List[int], xpath_subs_1: Optional[List[int]]=None) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        pad = [self.pad_xpath_subs_seq]
        if len(xpath_subs_1) == 0:
            return pad + xpath_subs_0 + pad
        return pad + xpath_subs_0 + pad + xpath_subs_1 + pad

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Args:\n        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding\n        special tokens using the tokenizer `prepare_for_model` method.\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n            already_has_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not the token list is already formatted with special tokens for the model.\n        Returns:\n            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.\n        '
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1, 1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not\n        make use of token type ids, therefore a list of zeros is returned.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n        Returns:\n            `List[int]`: List of zeros.\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + token_ids_1 + sep) * [0]

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(self, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]], text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]]=None, xpaths: Union[List[List[int]], List[List[List[int]]]]=None, node_labels: Optional[Union[List[int], List[List[int]]]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        if False:
            while True:
                i = 10
        '\n        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of\n        sequences with node-level xpaths and optional labels.\n\n        Args:\n            text (`str`, `List[str]`, `List[List[str]]`):\n                The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings\n                (nodes of a single example or questions of a batch of examples) or a list of list of strings (batch of\n                nodes).\n            text_pair (`List[str]`, `List[List[str]]`):\n                The sequence or batch of sequences to be encoded. Each sequence should be a list of strings\n                (pretokenized string).\n            xpaths (`List[List[int]]`, `List[List[List[int]]]`):\n                Node-level xpaths.\n            node_labels (`List[int]`, `List[List[int]]`, *optional*):\n                Node-level integer labels (for token classification tasks).\n        '

        def _is_valid_text_input(t):
            if False:
                print('Hello World!')
            if isinstance(t, str):
                return True
            elif isinstance(t, (list, tuple)):
                if len(t) == 0:
                    return True
                elif isinstance(t[0], str):
                    return True
                elif isinstance(t[0], (list, tuple)):
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False
        if text_pair is not None:
            if not _is_valid_text_input(text):
                raise ValueError('text input must of type `str` (single example) or `List[str]` (batch of examples). ')
            if not isinstance(text_pair, (list, tuple)):
                raise ValueError('Nodes must be of type `List[str]` (single pretokenized example), or `List[List[str]]` (batch of pretokenized examples).')
        elif not isinstance(text, (list, tuple)):
            raise ValueError('Nodes must be of type `List[str]` (single pretokenized example), or `List[List[str]]` (batch of pretokenized examples).')
        if text_pair is not None:
            is_batched = isinstance(text, (list, tuple))
        else:
            is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
        nodes = text if text_pair is None else text_pair
        assert xpaths is not None, 'You must provide corresponding xpaths'
        if is_batched:
            assert len(nodes) == len(xpaths), 'You must provide nodes and xpaths for an equal amount of examples'
            for (nodes_example, xpaths_example) in zip(nodes, xpaths):
                assert len(nodes_example) == len(xpaths_example), 'You must provide as many nodes as there are xpaths'
        else:
            assert len(nodes) == len(xpaths), 'You must provide as many nodes as there are xpaths'
        if is_batched:
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(f'batch length of `text`: {len(text)} does not match batch length of `text_pair`: {len(text_pair)}.')
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            is_pair = bool(text_pair is not None)
            return self.batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs, is_pair=is_pair, xpaths=xpaths, node_labels=node_labels, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)
        else:
            return self.encode_plus(text=text, text_pair=text_pair, xpaths=xpaths, node_labels=node_labels, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair], List[PreTokenizedInput]], is_pair: bool=None, xpaths: Optional[List[List[List[int]]]]=None, node_labels: Optional[Union[List[int], List[List[int]]]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        if False:
            for i in range(10):
                print('nop')
        (padding_strategy, truncation_strategy, max_length, kwargs) = self._get_padding_truncation_strategies(padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, verbose=verbose, **kwargs)
        return self._batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs, is_pair=is_pair, xpaths=xpaths, node_labels=node_labels, add_special_tokens=add_special_tokens, padding_strategy=padding_strategy, truncation_strategy=truncation_strategy, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)

    def _batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair], List[PreTokenizedInput]], is_pair: bool=None, xpaths: Optional[List[List[List[int]]]]=None, node_labels: Optional[List[List[int]]]=None, add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        if False:
            print('Hello World!')
        if return_offsets_mapping:
            raise NotImplementedError('return_offset_mapping is not available when using Python tokenizers. To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast.')
        batch_outputs = self._batch_prepare_for_model(batch_text_or_text_pairs=batch_text_or_text_pairs, is_pair=is_pair, xpaths=xpaths, node_labels=node_labels, add_special_tokens=add_special_tokens, padding_strategy=padding_strategy, truncation_strategy=truncation_strategy, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, return_tensors=return_tensors, verbose=verbose)
        return BatchEncoding(batch_outputs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(self, batch_text_or_text_pairs, is_pair: bool=None, xpaths: Optional[List[List[int]]]=None, node_labels: Optional[List[List[int]]]=None, add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[str]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_length: bool=False, verbose: bool=True) -> BatchEncoding:
        if False:
            i = 10
            return i + 15
        '\n        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It\n        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and\n        manages a moving window (with user defined stride) for overflowing tokens.\n\n        Args:\n            batch_ids_pairs: list of tokenized input ids or input ids pairs\n        '
        batch_outputs = {}
        for (idx, example) in enumerate(zip(batch_text_or_text_pairs, xpaths)):
            (batch_text_or_text_pair, xpaths_example) = example
            outputs = self.prepare_for_model(batch_text_or_text_pair[0] if is_pair else batch_text_or_text_pair, batch_text_or_text_pair[1] if is_pair else None, xpaths_example, node_labels=node_labels[idx] if node_labels is not None else None, add_special_tokens=add_special_tokens, padding=PaddingStrategy.DO_NOT_PAD.value, truncation=truncation_strategy.value, max_length=max_length, stride=stride, pad_to_multiple_of=None, return_attention_mask=False, return_token_type_ids=return_token_type_ids, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, return_tensors=None, prepend_batch_axis=False, verbose=verbose)
            for (key, value) in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        batch_outputs = self.pad(batch_outputs, padding=padding_strategy.value, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        return batch_outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING)
    def encode(self, text: Union[TextInput, PreTokenizedInput], text_pair: Optional[PreTokenizedInput]=None, xpaths: Optional[List[List[int]]]=None, node_labels: Optional[List[int]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        encoded_inputs = self.encode_plus(text=text, text_pair=text_pair, xpaths=xpaths, node_labels=node_labels, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)
        return encoded_inputs['input_ids']

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(self, text: Union[TextInput, PreTokenizedInput], text_pair: Optional[PreTokenizedInput]=None, xpaths: Optional[List[List[int]]]=None, node_labels: Optional[List[int]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        if False:
            for i in range(10):
                print('nop')
        '\n        Tokenize and prepare for the model a sequence or a pair of sequences. .. warning:: This method is deprecated,\n        `__call__` should be used instead.\n\n        Args:\n            text (`str`, `List[str]`, `List[List[str]]`):\n                The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.\n            text_pair (`List[str]` or `List[int]`, *optional*):\n                Optional second sequence to be encoded. This can be a list of strings (nodes of a single example) or a\n                list of list of strings (nodes of a batch of examples).\n        '
        (padding_strategy, truncation_strategy, max_length, kwargs) = self._get_padding_truncation_strategies(padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, verbose=verbose, **kwargs)
        return self._encode_plus(text=text, xpaths=xpaths, text_pair=text_pair, node_labels=node_labels, add_special_tokens=add_special_tokens, padding_strategy=padding_strategy, truncation_strategy=truncation_strategy, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)

    def _encode_plus(self, text: Union[TextInput, PreTokenizedInput], text_pair: Optional[PreTokenizedInput]=None, xpaths: Optional[List[List[int]]]=None, node_labels: Optional[List[int]]=None, add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        if False:
            return 10
        if return_offsets_mapping:
            raise NotImplementedError('return_offset_mapping is not available when using Python tokenizers. To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast. More information on available tokenizers at https://github.com/huggingface/transformers/pull/2674')
        return self.prepare_for_model(text=text, text_pair=text_pair, xpaths=xpaths, node_labels=node_labels, add_special_tokens=add_special_tokens, padding=padding_strategy.value, truncation=truncation_strategy.value, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, prepend_batch_axis=True, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_length=return_length, verbose=verbose)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(self, text: Union[TextInput, PreTokenizedInput], text_pair: Optional[PreTokenizedInput]=None, xpaths: Optional[List[List[int]]]=None, node_labels: Optional[List[int]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, prepend_batch_axis: bool=False, **kwargs) -> BatchEncoding:
        if False:
            i = 10
            return i + 15
        '\n        Prepares a sequence or a pair of sequences so that it can be used by the model. It adds special tokens,\n        truncates sequences if overflowing while taking into account the special tokens and manages a moving window\n        (with user defined stride) for overflowing tokens. Please Note, for *text_pair* different than `None` and\n        *truncation_strategy = longest_first* or `True`, it is not possible to return overflowing tokens. Such a\n        combination of arguments will raise an error.\n\n        Node-level `xpaths` are turned into token-level `xpath_tags_seq` and `xpath_subs_seq`. If provided, node-level\n        `node_labels` are turned into token-level `labels`. The node label is used for the first token of the node,\n        while remaining tokens are labeled with -100, such that they will be ignored by the loss function.\n\n        Args:\n            text (`str`, `List[str]`, `List[List[str]]`):\n                The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.\n            text_pair (`List[str]` or `List[int]`, *optional*):\n                Optional second sequence to be encoded. This can be a list of strings (nodes of a single example) or a\n                list of list of strings (nodes of a batch of examples).\n        '
        (padding_strategy, truncation_strategy, max_length, kwargs) = self._get_padding_truncation_strategies(padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, verbose=verbose, **kwargs)
        tokens = []
        pair_tokens = []
        xpath_tags_seq = []
        xpath_subs_seq = []
        pair_xpath_tags_seq = []
        pair_xpath_subs_seq = []
        labels = []
        if text_pair is None:
            if node_labels is None:
                for (word, xpath) in zip(text, xpaths):
                    if len(word) < 1:
                        continue
                    word_tokens = self.tokenize(word)
                    tokens.extend(word_tokens)
                    (xpath_tags_list, xpath_subs_list) = self.get_xpath_seq(xpath)
                    xpath_tags_seq.extend([xpath_tags_list] * len(word_tokens))
                    xpath_subs_seq.extend([xpath_subs_list] * len(word_tokens))
            else:
                for (word, xpath, label) in zip(text, xpaths, node_labels):
                    if len(word) < 1:
                        continue
                    word_tokens = self.tokenize(word)
                    tokens.extend(word_tokens)
                    (xpath_tags_list, xpath_subs_list) = self.get_xpath_seq(xpath)
                    xpath_tags_seq.extend([xpath_tags_list] * len(word_tokens))
                    xpath_subs_seq.extend([xpath_subs_list] * len(word_tokens))
                    if self.only_label_first_subword:
                        labels.extend([label] + [self.pad_token_label] * (len(word_tokens) - 1))
                    else:
                        labels.extend([label] * len(word_tokens))
        else:
            tokens = self.tokenize(text)
            xpath_tags_seq = [self.pad_xpath_tags_seq for _ in range(len(tokens))]
            xpath_subs_seq = [self.pad_xpath_subs_seq for _ in range(len(tokens))]
            for (word, xpath) in zip(text_pair, xpaths):
                if len(word) < 1:
                    continue
                word_tokens = self.tokenize(word)
                pair_tokens.extend(word_tokens)
                (xpath_tags_list, xpath_subs_list) = self.get_xpath_seq(xpath)
                pair_xpath_tags_seq.extend([xpath_tags_list] * len(word_tokens))
                pair_xpath_subs_seq.extend([xpath_subs_list] * len(word_tokens))
        ids = self.convert_tokens_to_ids(tokens)
        pair_ids = self.convert_tokens_to_ids(pair_tokens) if pair_tokens else None
        if return_overflowing_tokens and truncation_strategy == TruncationStrategy.LONGEST_FIRST and (pair_ids is not None):
            raise ValueError('Not possible to return overflowing tokens for pair of sequences with the `longest_first`. Please select another truncation strategy than `longest_first`, for instance `only_second` or `only_first`.')
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        overflowing_tokens = []
        overflowing_xpath_tags_seq = []
        overflowing_xpath_subs_seq = []
        overflowing_labels = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and (total_len > max_length):
            (ids, xpath_tags_seq, xpath_subs_seq, pair_ids, pair_xpath_tags_seq, pair_xpath_subs_seq, labels, overflowing_tokens, overflowing_xpath_tags_seq, overflowing_xpath_subs_seq, overflowing_labels) = self.truncate_sequences(ids, xpath_tags_seq=xpath_tags_seq, xpath_subs_seq=xpath_subs_seq, pair_ids=pair_ids, pair_xpath_tags_seq=pair_xpath_tags_seq, pair_xpath_subs_seq=pair_xpath_subs_seq, labels=labels, num_tokens_to_remove=total_len - max_length, truncation_strategy=truncation_strategy, stride=stride)
        if return_token_type_ids and (not add_special_tokens):
            raise ValueError('Asking to return token_type_ids while setting add_special_tokens to False results in an undefined behavior. Please set add_special_tokens to True or set return_token_type_ids to None.')
        if return_token_type_ids is None:
            return_token_type_ids = 'token_type_ids' in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = 'attention_mask' in self.model_input_names
        encoded_inputs = {}
        if return_overflowing_tokens:
            encoded_inputs['overflowing_tokens'] = overflowing_tokens
            encoded_inputs['overflowing_xpath_tags_seq'] = overflowing_xpath_tags_seq
            encoded_inputs['overflowing_xpath_subs_seq'] = overflowing_xpath_subs_seq
            encoded_inputs['overflowing_labels'] = overflowing_labels
            encoded_inputs['num_truncated_tokens'] = total_len - max_length
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
            xpath_tags_ids = self.build_xpath_tags_with_special_tokens(xpath_tags_seq, pair_xpath_tags_seq)
            xpath_subs_ids = self.build_xpath_subs_with_special_tokens(xpath_subs_seq, pair_xpath_subs_seq)
            if labels:
                labels = [self.pad_token_label] + labels + [self.pad_token_label]
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])
            xpath_tags_ids = xpath_tags_seq + pair_xpath_tags_seq if pair else xpath_tags_seq
            xpath_subs_ids = xpath_subs_seq + pair_xpath_subs_seq if pair else xpath_subs_seq
        encoded_inputs['input_ids'] = sequence
        encoded_inputs['xpath_tags_seq'] = xpath_tags_ids
        encoded_inputs['xpath_subs_seq'] = xpath_subs_ids
        if return_token_type_ids:
            encoded_inputs['token_type_ids'] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs['special_tokens_mask'] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs['special_tokens_mask'] = [0] * len(sequence)
        if labels:
            encoded_inputs['labels'] = labels
        self._eventual_warn_about_too_long_sequence(encoded_inputs['input_ids'], max_length, verbose)
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(encoded_inputs, max_length=max_length, padding=padding_strategy.value, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
        if return_length:
            encoded_inputs['length'] = len(encoded_inputs['input_ids'])
        batch_outputs = BatchEncoding(encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis)
        return batch_outputs

    def truncate_sequences(self, ids: List[int], xpath_tags_seq: List[List[int]], xpath_subs_seq: List[List[int]], pair_ids: Optional[List[int]]=None, pair_xpath_tags_seq: Optional[List[List[int]]]=None, pair_xpath_subs_seq: Optional[List[List[int]]]=None, labels: Optional[List[int]]=None, num_tokens_to_remove: int=0, truncation_strategy: Union[str, TruncationStrategy]='longest_first', stride: int=0) -> Tuple[List[int], List[int], List[int]]:
        if False:
            return 10
        "\n        Args:\n        Truncates a sequence pair in-place following the strategy.\n            ids (`List[int]`):\n                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and\n                `convert_tokens_to_ids` methods.\n            xpath_tags_seq (`List[List[int]]`):\n                XPath tag IDs of the first sequence.\n            xpath_subs_seq (`List[List[int]]`):\n                XPath sub IDs of the first sequence.\n            pair_ids (`List[int]`, *optional*):\n                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`\n                and `convert_tokens_to_ids` methods.\n            pair_xpath_tags_seq (`List[List[int]]`, *optional*):\n                XPath tag IDs of the second sequence.\n            pair_xpath_subs_seq (`List[List[int]]`, *optional*):\n                XPath sub IDs of the second sequence.\n            num_tokens_to_remove (`int`, *optional*, defaults to 0):\n                Number of tokens to remove using the truncation strategy.\n            truncation_strategy (`str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to\n            `False`):\n                The strategy to follow for truncation. Can be:\n                - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the\n                  maximum acceptable input length for the model if that argument is not provided. This will truncate\n                  token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a\n                  batch of pairs) is provided.\n                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the\n                  maximum acceptable input length for the model if that argument is not provided. This will only\n                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.\n                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the\n                  maximum acceptable input length for the model if that argument is not provided. This will only\n                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.\n                - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater\n                  than the model maximum admissible input size).\n            stride (`int`, *optional*, defaults to 0):\n                If set to a positive number, the overflowing tokens returned will contain some tokens from the main\n                sequence returned. The value of this argument defines the number of additional tokens.\n        Returns:\n            `Tuple[List[int], List[int], List[int]]`: The truncated `ids`, the truncated `pair_ids` and the list of\n            overflowing tokens. Note: The *longest_first* strategy returns empty list of overflowing tokens if a pair\n            of sequences (or a batch of pairs) is provided.\n        "
        if num_tokens_to_remove <= 0:
            return (ids, xpath_tags_seq, xpath_subs_seq, pair_ids, pair_xpath_tags_seq, pair_xpath_subs_seq, [], [], [])
        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)
        overflowing_tokens = []
        overflowing_xpath_tags_seq = []
        overflowing_xpath_subs_seq = []
        overflowing_labels = []
        if truncation_strategy == TruncationStrategy.ONLY_FIRST or (truncation_strategy == TruncationStrategy.LONGEST_FIRST and pair_ids is None):
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                overflowing_tokens = ids[-window_len:]
                overflowing_xpath_tags_seq = xpath_tags_seq[-window_len:]
                overflowing_xpath_subs_seq = xpath_subs_seq[-window_len:]
                ids = ids[:-num_tokens_to_remove]
                xpath_tags_seq = xpath_tags_seq[:-num_tokens_to_remove]
                xpath_subs_seq = xpath_subs_seq[:-num_tokens_to_remove]
                labels = labels[:-num_tokens_to_remove]
            else:
                error_msg = f'We need to remove {num_tokens_to_remove} to truncate the input but the first sequence has a length {len(ids)}. '
                if truncation_strategy == TruncationStrategy.ONLY_FIRST:
                    error_msg = error_msg + f"Please select another truncation strategy than {truncation_strategy}, for instance 'longest_first' or 'only_second'."
                logger.error(error_msg)
        elif truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            logger.warning(f"Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the '{TruncationStrategy.LONGEST_FIRST.value}' truncation strategy. So the returned list will always be empty even if some tokens have been removed.")
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    ids = ids[:-1]
                    xpath_tags_seq = xpath_tags_seq[:-1]
                    xpath_subs_seq = xpath_subs_seq[:-1]
                    labels = labels[:-1]
                else:
                    pair_ids = pair_ids[:-1]
                    pair_xpath_tags_seq = pair_xpath_tags_seq[:-1]
                    pair_xpath_subs_seq = pair_xpath_subs_seq[:-1]
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                overflowing_tokens = pair_ids[-window_len:]
                overflowing_xpath_tags_seq = pair_xpath_tags_seq[-window_len:]
                overflowing_xpath_subs_seq = pair_xpath_subs_seq[-window_len:]
                pair_ids = pair_ids[:-num_tokens_to_remove]
                pair_xpath_tags_seq = pair_xpath_tags_seq[:-num_tokens_to_remove]
                pair_xpath_subs_seq = pair_xpath_subs_seq[:-num_tokens_to_remove]
            else:
                logger.error(f"We need to remove {num_tokens_to_remove} to truncate the input but the second sequence has a length {len(pair_ids)}. Please select another truncation strategy than {truncation_strategy}, for instance 'longest_first' or 'only_first'.")
        return (ids, xpath_tags_seq, xpath_subs_seq, pair_ids, pair_xpath_tags_seq, pair_xpath_subs_seq, labels, overflowing_tokens, overflowing_xpath_tags_seq, overflowing_xpath_subs_seq, overflowing_labels)

    def _pad(self, encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding], max_length: Optional[int]=None, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None) -> dict:
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)\n            encoded_inputs:\n                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).\n            max_length: maximum length of the returned list and optionally padding length (see below).\n                Will truncate by taking into account the special tokens.\n            padding_strategy: PaddingStrategy to use for padding.\n                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch\n                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)\n                - PaddingStrategy.DO_NOT_PAD: Do not pad\n                The tokenizer padding sides are defined in self.padding_side:\n                    - 'left': pads on the left of the sequences\n                    - 'right': pads on the right of the sequences\n            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.\n                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability\n                `>= 7.5` (Volta).\n            return_attention_mask:\n                (optional) Set to False to avoid returning attention mask (default: set to model specifics)\n        "
        if return_attention_mask is None:
            return_attention_mask = 'attention_mask' in self.model_input_names
        required_input = encoded_inputs[self.model_input_names[0]]
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
        if return_attention_mask and 'attention_mask' not in encoded_inputs:
            encoded_inputs['attention_mask'] = [1] * len(required_input)
        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs['attention_mask'] = encoded_inputs['attention_mask'] + [0] * difference
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = encoded_inputs['token_type_ids'] + [self.pad_token_type_id] * difference
                if 'xpath_tags_seq' in encoded_inputs:
                    encoded_inputs['xpath_tags_seq'] = encoded_inputs['xpath_tags_seq'] + [self.pad_xpath_tags_seq] * difference
                if 'xpath_subs_seq' in encoded_inputs:
                    encoded_inputs['xpath_subs_seq'] = encoded_inputs['xpath_subs_seq'] + [self.pad_xpath_subs_seq] * difference
                if 'labels' in encoded_inputs:
                    encoded_inputs['labels'] = encoded_inputs['labels'] + [self.pad_token_label] * difference
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = encoded_inputs['special_tokens_mask'] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs['attention_mask'] = [0] * difference + encoded_inputs['attention_mask']
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = [self.pad_token_type_id] * difference + encoded_inputs['token_type_ids']
                if 'xpath_tags_seq' in encoded_inputs:
                    encoded_inputs['xpath_tags_seq'] = [self.pad_xpath_tags_seq] * difference + encoded_inputs['xpath_tags_seq']
                if 'xpath_subs_seq' in encoded_inputs:
                    encoded_inputs['xpath_subs_seq'] = [self.pad_xpath_subs_seq] * difference + encoded_inputs['xpath_subs_seq']
                if 'labels' in encoded_inputs:
                    encoded_inputs['labels'] = [self.pad_token_label] * difference + encoded_inputs['labels']
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = [1] * difference + encoded_inputs['special_tokens_mask']
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError('Invalid padding strategy:' + str(self.padding_side))
        return encoded_inputs