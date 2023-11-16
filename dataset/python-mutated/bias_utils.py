import torch
import json
from os import PathLike
from typing import List, Tuple, Union, Optional
from allennlp.common.file_utils import cached_path
from allennlp.data import Vocabulary
from allennlp.data.tokenizers.tokenizer import Tokenizer

def _convert_word_to_ids_tensor(word, tokenizer, vocab, namespace, all_cases):
    if False:
        while True:
            i = 10
    if all_cases:
        words_list = [word.lower(), word.title(), word.upper()]
    else:
        words_list = [word]
    ids = []
    for w in words_list:
        if vocab:
            tokens = tokenizer.tokenize(w)
            ids.append(torch.tensor([vocab.get_token_index(t.text, namespace) for t in tokens]))
        else:
            ids.append(torch.tensor(tokenizer.tokenizer(w)['input_ids']))
    return ids

def load_words(fname: Union[str, PathLike], tokenizer: Tokenizer, vocab: Optional[Vocabulary]=None, namespace: str='tokens', all_cases: bool=True) -> List[torch.Tensor]:
    if False:
        i = 10
        return i + 15
    "\n    This function loads a list of words from a file,\n    tokenizes each word into subword tokens, and converts the\n    tokens into IDs.\n\n    # Parameters\n\n    fname : `Union[str, PathLike]`\n        Name of file containing list of words to load.\n    tokenizer : `Tokenizer`\n        Tokenizer to tokenize words in file.\n    vocab : `Vocabulary`, optional (default=`None`)\n        Vocabulary of tokenizer. If `None`, assumes tokenizer is of\n        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.\n    namespace : `str`\n        Namespace of vocab to use when tokenizing.\n    all_cases : `bool`, optional (default=`True`)\n        Whether to tokenize lower, title, and upper cases of each word.\n\n    # Returns\n\n    word_ids : `List[torch.Tensor]`\n        List of tensors containing the IDs of subword tokens for\n        each word in the file.\n    "
    word_ids = []
    with open(cached_path(fname)) as f:
        words = json.load(f)
        for w in words:
            word_ids.extend(_convert_word_to_ids_tensor(w, tokenizer, vocab, namespace, all_cases))
    return word_ids

def load_word_pairs(fname: Union[str, PathLike], tokenizer: Tokenizer, vocab: Optional[Vocabulary]=None, namespace: str='token', all_cases: bool=True) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if False:
        while True:
            i = 10
    "\n    This function loads a list of pairs of words from a file,\n    tokenizes each word into subword tokens, and converts the\n    tokens into IDs.\n\n    # Parameters\n\n    fname : `Union[str, PathLike]`\n        Name of file containing list of pairs of words to load.\n    tokenizer : `Tokenizer`\n        Tokenizer to tokenize words in file.\n    vocab : `Vocabulary`, optional (default=`None`)\n        Vocabulary of tokenizer. If `None`, assumes tokenizer is of\n        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.\n    namespace : `str`\n        Namespace of vocab to use when tokenizing.\n    all_cases : `bool`, optional (default=`True`)\n        Whether to tokenize lower, title, and upper cases of each word.\n\n    # Returns\n\n    word_ids : `Tuple[List[torch.Tensor], List[torch.Tensor]]`\n        Pair of lists of tensors containing the IDs of subword tokens for\n        words in the file.\n    "
    word_ids1 = []
    word_ids2 = []
    with open(cached_path(fname)) as f:
        words = json.load(f)
        for (w1, w2) in words:
            word_ids1.extend(_convert_word_to_ids_tensor(w1, tokenizer, vocab, namespace, all_cases))
            word_ids2.extend(_convert_word_to_ids_tensor(w2, tokenizer, vocab, namespace, all_cases))
    return (word_ids1, word_ids2)