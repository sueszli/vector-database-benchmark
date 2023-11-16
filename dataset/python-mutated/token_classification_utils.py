"""
Helper methods used internally in cleanlab.token_classification
"""
from __future__ import annotations
import re
import string
import numpy as np
from termcolor import colored
from typing import List, Optional, Callable, Tuple, TypeVar, TYPE_CHECKING
if TYPE_CHECKING:
    import numpy.typing as npt
    T = TypeVar('T', bound=npt.NBitBase)

def get_sentence(words: List[str]) -> str:
    if False:
        print('Hello World!')
    '\n    Get sentence formed by a list of words with minor processing for readability\n\n    Parameters\n    ----------\n    words:\n        list of word-level tokens\n\n    Returns\n    ----------\n    sentence:\n        sentence formed by list of word-level tokens\n\n    Examples\n    --------\n    >>> from cleanlab.internal.token_classification_utils import get_sentence\n    >>> words = ["This", "is", "a", "sentence", "."]\n    >>> get_sentence(words)\n    \'This is a sentence.\'\n    '
    sentence = ''
    for word in words:
        if word not in string.punctuation or word in ['-', '(']:
            word = ' ' + word
        sentence += word
    sentence = sentence.replace(" '", "'").replace('( ', '(').strip()
    return sentence

def filter_sentence(sentences: List[str], condition: Optional[Callable[[str], bool]]=None) -> Tuple[List[str], List[bool]]:
    if False:
        i = 10
        return i + 15
    '\n    Filter sentence based on some condition, and returns filter mask\n\n    Parameters\n    ----------\n    sentences:\n        list of sentences\n\n    condition:\n        sentence filtering condition\n\n    Returns\n    ---------\n    sentences:\n        list of sentences filtered\n\n    mask:\n        boolean mask such that `mask[i] == True` if the i\'th sentence is included in the\n        filtered sentence, otherwise `mask[i] == False`\n\n    Examples\n    --------\n    >>> from cleanlab.internal.token_classification_utils import filter_sentence\n    >>> sentences = ["Short sentence.", "This is a longer sentence."]\n    >>> condition = lambda x: len(x.split()) > 2\n    >>> long_sentences, _ = filter_sentence(sentences, condition)\n    >>> long_sentences\n    [\'This is a longer sentence.\']\n    >>> document = ["# Headline", "Sentence 1.", "&", "Sentence 2."]\n    >>> sentences, mask = filter_sentence(document)\n    >>> sentences, mask\n    ([\'Sentence 1.\', \'Sentence 2.\'], [False, True, False, True])\n    '
    if not condition:
        condition = lambda sentence: len(sentence) > 1 and '#' not in sentence
    mask = list(map(condition, sentences))
    sentences = [sentence for (m, sentence) in zip(mask, sentences) if m]
    return (sentences, mask)

def process_token(token: str, replace: List[Tuple[str, str]]=[('#', '')]) -> str:
    if False:
        return 10
    '\n    Replaces special characters in the tokens\n\n    Parameters\n    ----------\n    token:\n        token which potentially contains special characters\n\n    replace:\n        list of tuples `(s1, s2)`, where all occurances of s1 are replaced by s2\n\n    Returns\n    ---------\n    processed_token:\n        processed token whose special character has been replaced\n\n    Note\n    ----\n        Only applies to characters in the original input token.\n\n    Examples\n    --------\n    >>> from cleanlab.internal.token_classification_utils import process_token\n    >>> token = "#Comment"\n    >>> process_token("#Comment")\n    \'Comment\'\n\n    Specify custom replacement rules\n\n    >>> replace = [("C", "a"), ("a", "C")]\n    >>> process_token("Cleanlab", replace)\n    \'aleCnlCb\'\n    '
    replace_dict = {re.escape(k): v for (k, v) in replace}
    pattern = '|'.join(replace_dict.keys())
    compiled_pattern = re.compile(pattern)
    replacement = lambda match: replace_dict[re.escape(match.group(0))]
    processed_token = compiled_pattern.sub(replacement, token)
    return processed_token

def mapping(entities: List[int], maps: List[int]) -> List[int]:
    if False:
        return 10
    '\n    Map a list of entities to its corresponding entities\n\n    Parameters\n    ----------\n    entities:\n        a list of given entities\n\n    maps:\n        a list of mapped entities, such that the i\'th indexed token should be mapped to `maps[i]`\n\n    Returns\n    ---------\n    mapped_entities:\n        a list of mapped entities\n\n    Examples\n    --------\n    >>> unique_identities = [0, 1, 2, 3, 4]  # ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]\n    >>> maps = [0, 1, 1, 2, 2]  # ["O", "PER", "PER", "LOC", "LOC"]\n    >>> mapping(unique_identities, maps)\n    [0, 1, 1, 2, 2]  # ["O", "PER", "PER", "LOC", "LOC"]\n    >>> mapping([0, 0, 4, 4, 3, 4, 0, 2], maps)\n    [0, 0, 2, 2, 2, 2, 0, 1]  # ["O", "O", "LOC", "LOC", "LOC", "LOC", "O", "PER"]\n    '
    f = lambda x: maps[x]
    return list(map(f, entities))

def merge_probs(probs: npt.NDArray['np.floating[T]'], maps: List[int]) -> npt.NDArray['np.floating[T]']:
    if False:
        return 10
    "\n    Merges model-predictive probabilities with desired mapping\n\n    Parameters\n    ----------\n    probs:\n        A 2D np.array of shape `(N, K)`, where N is the number of tokens, and K is the number of classes for the model\n\n    maps:\n        a list of mapped index, such that the probability of the token being in the i'th class is mapped to the\n        `maps[i]` index. If `maps[i] == -1`, the i'th column of `probs` is ignored. If `np.any(maps == -1)`, the\n        returned probability is re-normalized.\n\n    Returns\n    ---------\n    probs_merged:\n        A 2D np.array of shape ``(N, K')``, where `K'` is the number of new classes. Probabilities are merged and\n        re-normalized if necessary.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from cleanlab.internal.token_classification_utils import merge_probs\n    >>> probs = np.array([\n    ...     [0.55, 0.0125, 0.0375, 0.1, 0.3],\n    ...     [0.1, 0.8, 0, 0.075, 0.025],\n    ... ])\n    >>> maps = [0, 1, 1, 2, 2]\n    >>> merge_probs(probs, maps)\n    array([[0.55, 0.05, 0.4 ],\n           [0.1 , 0.8 , 0.1 ]])\n    "
    old_classes = probs.shape[1]
    map_size = np.max(maps) + 1
    probs_merged = np.zeros([len(probs), map_size], dtype=probs.dtype.type)
    for i in range(old_classes):
        if maps[i] >= 0:
            probs_merged[:, maps[i]] += probs[:, i]
    if -1 in maps:
        row_sums = probs_merged.sum(axis=1)
        probs_merged /= row_sums[:, np.newaxis]
    return probs_merged

def color_sentence(sentence: str, word: str) -> str:
    if False:
        print('Hello World!')
    '\n    Searches for a given token in the sentence and returns the sentence where the given token is colored red\n\n    Parameters\n    ----------\n    sentence:\n        a sentence where the word is searched\n\n    word:\n        keyword to find in `sentence`. Assumes the word exists in the sentence.\n    Returns\n    ---------\n    colored_sentence:\n        `sentence` where the every occurrence of the word is colored red, using ``termcolor.colored``\n\n    Examples\n    --------\n    >>> from cleanlab.internal.token_classification_utils import color_sentence\n    >>> sentence = "This is a sentence."\n    >>> word = "sentence"\n    >>> color_sentence(sentence, word)\n    \'This is a \x1b[31msentence\x1b[0m.\'\n\n    Also works for multiple occurrences of the word\n\n    >>> document = "This is a sentence. This is another sentence."\n    >>> word = "sentence"\n    >>> color_sentence(document, word)\n    \'This is a \x1b[31msentence\x1b[0m. This is another \x1b[31msentence\x1b[0m.\'\n    '
    colored_word = colored(word, 'red')
    return _replace_sentence(sentence=sentence, word=word, new_word=colored_word)

def _replace_sentence(sentence: str, word: str, new_word: str) -> str:
    if False:
        return 10
    '\n    Searches for a given token in the sentence and returns the sentence where the given token has been replaced by\n    `new_word`.\n\n    Parameters\n    ----------\n    sentence:\n        a sentence where the word is searched\n\n    word:\n        keyword to find in `sentence`. Assumes the word exists in the sentence.\n\n    new_word:\n        the word to replace the keyword with\n\n    Returns\n    ---------\n    new_sentence:\n        `sentence` where the every occurrence of the word is replaced by `colored_word`\n    '
    (new_sentence, number_of_substitions) = re.subn('\\b{}\\b'.format(re.escape(word)), new_word, sentence)
    if number_of_substitions == 0:
        new_sentence = sentence.replace(word, new_word)
    return new_sentence