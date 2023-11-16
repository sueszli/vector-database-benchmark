import logging
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
logger = logging.getLogger()

def _get_matching_blocks_native(query, text):
    if False:
        i = 10
        return i + 15
    return SequenceMatcher(None, query, text).get_matching_blocks()
try:
    from Levenshtein import editops, matching_blocks

    def _get_matching_blocks(query, text):
        if False:
            print('Hello World!')
        return matching_blocks(editops(query, text), query, text)
except ImportError:
    logger.warning('Levenshtein is missing or outdated. Falling back to slower fuzzy-finding method.')
    _get_matching_blocks = _get_matching_blocks_native

def _normalize(string):
    if False:
        while True:
            i = 10
    return unicodedata.normalize('NFD', string.casefold()).encode('ascii', 'ignore').decode('utf-8')

@lru_cache(maxsize=1000)
def get_matching_blocks(query, text):
    if False:
        print('Hello World!')
    '\n    Uses our _get_matching_blocks wrapper method to find the blocks using "Longest Common Substrings",\n    :returns: list of tuples, containing the index and matching block, number of characters that matched\n    '
    blocks = _get_matching_blocks(_normalize(query), _normalize(text))[:-1]
    output = []
    total_len = 0
    for (_, text_index, length) in blocks:
        output.append((text_index, text[text_index:text_index + length]))
        total_len += length
    return (output, total_len)

def get_score(query, text):
    if False:
        i = 10
        return i + 15
    '\n    Uses get_matching_blocks() to figure out how much of the query that matches the text,\n    and tries to weight this to slightly favor shorter results and largely favor word matches\n    :returns: number between 0 and 100\n    '
    if not query or not text:
        return 0
    query_len = len(query)
    text_len = len(text)
    max_len = max(query_len, text_len)
    (blocks, matching_chars) = get_matching_blocks(query, text)
    base_similarity = matching_chars / query_len
    for (index, _) in blocks:
        is_word_boundary = index == 0 or text[index - 1] == ' '
        if not is_word_boundary:
            base_similarity -= 0.5 / query_len
    return 100 * base_similarity * query_len / (query_len + (max_len - query_len) * 0.001)