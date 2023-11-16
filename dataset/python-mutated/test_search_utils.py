import time
from collections import deque
import pytest
from tribler.core.utilities.search_utils import filter_keywords, find_word_and_rotate_title, freshness_rank, item_rank, seeders_rank, split_into_keywords, title_rank, torrent_rank
DAY = 60 * 60 * 24

def test_split_into_keywords():
    if False:
        while True:
            i = 10
    result = split_into_keywords('to be or not to be')
    assert isinstance(result, list)
    assert len(result) == 6
    result = split_into_keywords('to be or not to be', True)
    assert isinstance(result, list)
    assert len(result) == 4

def test_filter_keywords():
    if False:
        for i in range(10):
            print('nop')
    result = filter_keywords(['to', 'be', 'or', 'not', 'to', 'be'])
    assert isinstance(result, list)
    assert len(result) == 4

def test_title_rank_range():
    if False:
        for i in range(10):
            print('nop')
    assert title_rank('Big Buck Bunny', 'Big Buck Bunny') == 1
    long_query = ' '.join(['foo'] * 1000)
    long_title = ' '.join(['bar'] * 1000)
    assert title_rank(long_query, long_title) == pytest.approx(0.03554968)

def test_freshness_rank_range():
    if False:
        while True:
            i = 10
    assert freshness_rank(-1) == freshness_rank(None) == 0
    assert freshness_rank(0) == 1
    assert freshness_rank(0.001) == pytest.approx(1.0)
    assert freshness_rank(1000000000) == pytest.approx(0.0025852989)

def test_seeders_rank_range():
    if False:
        return 10
    assert seeders_rank(0) == 0
    assert seeders_rank(1000000) == pytest.approx(0.9999)

def test_torrent_rank_range():
    if False:
        return 10
    assert torrent_rank('Big Buck Bunny', 'Big Buck Bunny', seeders=1000000, freshness=0.01) == pytest.approx(0.99999)
    long_query = ' '.join(['foo'] * 1000)
    long_title = ' '.join(['bar'] * 1000)
    assert torrent_rank(long_query, long_title, freshness=1000000 * 365 * DAY) == pytest.approx(+0.02879524)

def test_torrent_rank():
    if False:
        return 10
    query = 'Big Buck Bunny'
    title_match = torrent_rank(query, 'Big Buck Bunny')
    assert title_match > 0.8
    assert torrent_rank(query, 'Big Buck Bunny', seeders=1000, freshness=1 * DAY) > torrent_rank(query, 'Big Buck Bunny', seeders=1000, freshness=100 * DAY) > torrent_rank(query, 'Big Buck Bunny', seeders=100, freshness=100 * DAY) > title_match
    assert title_match > torrent_rank(query, 'Big Buck Bunny II') > torrent_rank(query, 'Big Buck Brown Bunny') > torrent_rank(query, 'Big Bad Buck Bunny') > torrent_rank(query, 'Boring Big Buck Bunny')
    assert title_match > torrent_rank(query, 'Big Buck A Bunny') > torrent_rank(query, 'Big Buck A B Bunny') > torrent_rank(query, 'Big Buck A B C Bunny')
    assert title_match > torrent_rank(query, 'Big A Buck Bunny') > torrent_rank(query, 'Big A B Buck Bunny') > torrent_rank(query, 'Big A B C Buck Bunny')
    assert title_match > torrent_rank(query, 'A Big Buck Bunny') > torrent_rank(query, 'A B Big Buck Bunny') > torrent_rank(query, 'A B C Big Buck Bunny')
    assert torrent_rank(query, 'Big A Buck Bunny') > torrent_rank(query, 'A Big Buck Bunny')
    assert torrent_rank(query, 'Big A B Buck Bunny') > torrent_rank(query, 'A B Big Buck Bunny')
    assert torrent_rank(query, 'Big A B C Buck Bunny') > torrent_rank(query, 'A B C Big Buck Bunny')
    assert title_match > torrent_rank(query, 'Big Bunny Buck')
    assert torrent_rank(query, 'Big Buck') < 0.5
    assert torrent_rank(query, 'Big Buck') > torrent_rank(query, 'Big Bunny') > torrent_rank(query, 'Buck Bunny')
    assert torrent_rank(query, 'Buck Bunny', seeders=1000, freshness=5 * DAY) > torrent_rank(query, 'Buck Bunny', seeders=100, freshness=5 * DAY) > torrent_rank(query, 'Buck Bunny', seeders=10, freshness=5 * DAY) > torrent_rank(query, 'Buck Bunny')
    assert torrent_rank(query, 'Buck Bunny', freshness=5 * DAY) > torrent_rank(query, 'Buck Bunny', freshness=10 * DAY) > torrent_rank(query, 'Buck Bunny', freshness=20 * DAY)
    assert torrent_rank('Sintel', 'Sintel') > 0.8
    assert torrent_rank('Sintel', 'Sintel') > torrent_rank('Sintel', 'Sintel Part II') > torrent_rank('Sintel', 'Part of Sintel') > torrent_rank('Sintel', 'the.script.from.the.movie.Sintel.pdf')
    assert torrent_rank("Internet's Own Boy", "Internet's Own Boy") > torrent_rank("Internet's Own Boy", "Internet's very Own Boy") > torrent_rank("Internet's Own Boy", "Internet's very special Boy person")

def test_title_rank():
    if False:
        return 10
    assert title_rank('', 'title') == pytest.approx(1.0)
    assert title_rank('query', '') == pytest.approx(0.0)

def test_item_rank():
    if False:
        return 10
    item = dict(name='abc', num_seeders=10, num_leechers=20, created=time.time() - 10 * DAY)
    assert item_rank('abc', item) == pytest.approx(0.88794642)
    item = dict(name='abc', num_seeders=10, num_leechers=20, created=0)
    assert item_rank('abc', item) == pytest.approx(0.81964285)
    item = dict(name='abc', num_seeders=10, num_leechers=20)
    assert item_rank('abc', item) == pytest.approx(0.81964285)

def test_find_word():
    if False:
        return 10
    title = deque(['A', 'B', 'C'])
    assert find_word_and_rotate_title('A', title) == (True, 0) and title == deque(['B', 'C'])
    assert find_word_and_rotate_title('B', title) == (True, 0) and title == deque(['C'])
    assert find_word_and_rotate_title('C', title) == (True, 0) and title == deque([])
    title = deque(['A', 'B', 'C', 'D'])
    assert find_word_and_rotate_title('A', title) == (True, 0) and title == deque(['B', 'C', 'D'])
    assert find_word_and_rotate_title('B', title) == (True, 0) and title == deque(['C', 'D'])
    assert find_word_and_rotate_title('C', title) == (True, 0) and title == deque(['D'])
    title = deque(['X', 'Y', 'A', 'B', 'C'])
    assert find_word_and_rotate_title('A', title) == (True, 2) and title == deque(['B', 'C', 'X', 'Y'])
    assert find_word_and_rotate_title('B', title) == (True, 0) and title == deque(['C', 'X', 'Y'])
    assert find_word_and_rotate_title('C', title) == (True, 0) and title == deque(['X', 'Y'])
    title = deque(['A', 'B', 'X', 'Y', 'C'])
    assert find_word_and_rotate_title('A', title) == (True, 0) and title == deque(['B', 'X', 'Y', 'C'])
    assert find_word_and_rotate_title('B', title) == (True, 0) and title == deque(['X', 'Y', 'C'])
    assert find_word_and_rotate_title('C', title) == (True, 2) and title == deque(['X', 'Y'])
    title = deque(['A', 'C', 'B'])
    assert find_word_and_rotate_title('A', title) == (True, 0) and title == deque(['C', 'B'])
    assert find_word_and_rotate_title('B', title) == (True, 1) and title == deque(['C'])
    assert find_word_and_rotate_title('C', title) == (True, 0) and title == deque([])
    title = deque(['A', 'C', 'X'])
    assert find_word_and_rotate_title('A', title) == (True, 0) and title == deque(['C', 'X'])
    assert find_word_and_rotate_title('B', title) == (False, 0) and title == deque(['C', 'X'])
    assert find_word_and_rotate_title('C', title) == (True, 0) and title == deque(['X'])