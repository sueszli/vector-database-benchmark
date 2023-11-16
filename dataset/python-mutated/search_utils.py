"""
Search utilities.

Author(s): Jelle Roozenburg, Arno Bakker, Alexander Kozlovsky
"""
import re
import time
from collections import deque
from typing import Deque, List, Optional, Tuple
RE_KEYWORD_SPLIT = re.compile('[\\W_]', re.UNICODE)
DIALOG_STOPWORDS = {'an', 'and', 'by', 'for', 'from', 'of', 'the', 'to', 'with'}
SECONDS_IN_DAY = 60 * 60 * 24

def split_into_keywords(string, to_filter_stopwords=False):
    if False:
        i = 10
        return i + 15
    '\n    Takes a (unicode) string and returns a list of (unicode) lowercase\n    strings.  No empty strings are returned.\n\n    We currently split on non-alphanumeric characters and the\n    underscore.\n\n    If to_filter_stopwords is True a small stopword filter is using to reduce the number of keywords\n    '
    if to_filter_stopwords:
        return [kw for kw in RE_KEYWORD_SPLIT.split(string.lower()) if len(kw) > 0 and kw not in DIALOG_STOPWORDS]
    else:
        return [kw for kw in RE_KEYWORD_SPLIT.split(string.lower()) if len(kw) > 0]

def filter_keywords(keywords):
    if False:
        while True:
            i = 10
    return [kw for kw in keywords if len(kw) > 0 and kw not in DIALOG_STOPWORDS]

def item_rank(query: str, item: dict) -> float:
    if False:
        print('Hello World!')
    '\n    Calculates the torrent rank for item received from remote query. Returns the torrent rank value in range [0, 1].\n\n    :param query: a user-defined query string\n    :param item: a dict with torrent info.\n                 Should include key `name`, can include `num_seeders`, `num_leechers`, `created`\n    :return: the torrent rank value in range [0, 1]\n    '
    title = item['name']
    seeders = item.get('num_seeders', 0)
    leechers = item.get('num_leechers', 0)
    created = item.get('created', 0)
    freshness = None if created <= 0 else time.time() - created
    return torrent_rank(query, title, seeders, leechers, freshness)

def torrent_rank(query: str, title: str, seeders: int=0, leechers: int=0, freshness: Optional[float]=None) -> float:
    if False:
        while True:
            i = 10
    '\n    Calculates search rank for a torrent.\n\n    :param query: a user-defined query string\n    :param title: a torrent name\n    :param seeders: the number of seeders\n    :param leechers: the number of leechers\n    :param freshness: the number of seconds since the torrent creation. Zero or negative value means the torrent\n                      creation date is unknown. It is more convenient to use comparing to a timestamp, as it avoids\n                      using the `time()` function call and simplifies testing.\n    :return: the torrent rank value in range [0, 1]\n\n    Takes into account:\n      - similarity of the title to the query string;\n      - the reported number of seeders;\n      - how long ago the torrent file was created.\n    '
    tr = title_rank(query or '', title or '')
    sr = (seeders_rank(seeders or 0, leechers or 0) + 9) / 10
    fr = (freshness_rank(freshness) + 9) / 10
    result = tr * sr * fr
    return result
LEECHERS_COEFF = 0.1
SEEDERS_HALF_RANK = 100

def seeders_rank(seeders: int, leechers: int=0) -> float:
    if False:
        print('Hello World!')
    "\n    Calculates rank based on the number of torrent's seeders and leechers\n\n    :param seeders: the number of seeders for the torrent. It is a positive value, usually in the range [0, 1000]\n    :param leechers: the number of leechers for the torrent. It is a positive value, usually in the range [0, 1000]\n    :return: the torrent rank based on seeders and leechers, normalized to the range [0, 1]\n    "
    sl = seeders + leechers * LEECHERS_COEFF
    return sl / (100 + sl)

def freshness_rank(freshness: Optional[float]) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates a rank value based on the torrent freshness. The result is normalized to the range [0, 1]\n\n    :param freshness: number of seconds since the torrent creation.\n                      None means the actual torrent creation date is unknown.\n                      Negative values treated as invalid values and give the same result as None\n    :return: the torrent rank based on freshness. The result is normalized to the range [0, 1]\n\n    Example results:\n    0 seconds since torrent creation -> the actual torrent creation date is unknown, freshness rank 0\n    1 second since torrent creation -> freshness rank 0.999\n    1 day since torrent creation -> freshness rank 0.967\n    30 days since torrent creation -> freshness rank 0.5\n    1 year since torrent creation -> freshness rank 0.0759\n    '
    if freshness is None or freshness < 0:
        return 0
    days = (freshness or 0) / SECONDS_IN_DAY
    return 1 / (1 + days / 30)
word_re = re.compile('\\w+', re.UNICODE)

def title_rank(query: str, title: str) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate the similarity of the title string to a query string as a float value in range [0, 1]\n\n    :param query: a user-defined query string\n    :param title: a torrent name\n    :return: the similarity of the title string to a query string as a float value in range [0, 1]\n    '
    query = word_re.findall(query.lower())
    title = word_re.findall(title.lower())
    return calculate_rank(query, title)
POSITION_COEFF = 5
MISSED_WORD_PENALTY = 10
REMAINDER_COEFF = 10
RANK_NORMALIZATION_COEFF = 10

def calculate_rank(query: List[str], title: List[str]) -> float:
    if False:
        while True:
            i = 10
    '\n    Calculates the similarity of the title to the query as a float value in range [0, 1].\n\n    :param query: list of query words\n    :param title: list of title words\n    :return: the similarity of the title to the query as a float value in range [0, 1]\n    '
    if not query:
        return 1.0
    if not title:
        return 0.0
    title = deque(title)
    total_error = 0
    for (i, word) in enumerate(query):
        word_weight = POSITION_COEFF / (POSITION_COEFF + i)
        (found, skipped) = find_word_and_rotate_title(word, title)
        if found:
            total_error += skipped * word_weight
        else:
            total_error += MISSED_WORD_PENALTY * word_weight
    remainder_weight = 1 / (REMAINDER_COEFF + len(query))
    remained_words_error = len(title) * remainder_weight
    total_error += remained_words_error
    return RANK_NORMALIZATION_COEFF / (RANK_NORMALIZATION_COEFF + total_error)

def find_word_and_rotate_title(word: str, title: Deque[str]) -> Tuple[bool, int]:
    if False:
        return 10
    "\n    Finds the query word in the title. Returns whether it was found or not and the number of skipped words in the title.\n\n    :param word: a word from the user-defined query string\n    :param title: a deque of words in the title\n    :return: a two-elements tuple, whether the word was found in the title and the number of skipped words\n\n    This is a helper function to efficiently answer a question of how close a query string and a title string are,\n    taking into account the ordering of words in both strings.\n\n    For efficiency reasons, the function modifies the `title` deque in place by removing the first entrance\n    of the found word and rotating all leading non-matching words to the end of the deque. It allows to efficiently\n    perform multiple calls of the `find_word_and_rotate_title` function for subsequent words from the same query string.\n\n    An example: find_word_and_rotate_title('A', deque(['X', 'Y', 'A', 'B', 'C'])) returns `(True, 2)`, where True means\n    that the word 'A' was found in the `title` deque, and 2 is the number of skipped words ('X', 'Y'). Also, it modifies\n    the `title` deque, so it starts looking like deque(['B', 'C', 'X', 'Y']). The found word 'A' was removed, and\n    the leading non-matching words ('X', 'Y') were moved to the end of the deque.\n    "
    try:
        skipped = title.index(word)
    except ValueError:
        return (False, 0)
    title.rotate(-skipped)
    title.popleft()
    return (True, skipped)