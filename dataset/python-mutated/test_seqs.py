from collections.abc import Iterator
from operator import add
import pytest
from whatever import _
from funcy import is_list
from funcy.seqs import *

def test_repeatedly():
    if False:
        print('Hello World!')
    counter = count()
    c = lambda : next(counter)
    assert take(2, repeatedly(c)) == [0, 1]

def test_iterate():
    if False:
        i = 10
        return i + 15
    assert take(4, iterate(_ * 2, 1)) == [1, 2, 4, 8]

def test_take():
    if False:
        return 10
    assert take(2, [3, 2, 1]) == [3, 2]
    assert take(2, count(7)) == [7, 8]

def test_drop():
    if False:
        while True:
            i = 10
    dropped = drop(2, [5, 4, 3, 2])
    assert isinstance(dropped, Iterator)
    assert list(dropped) == [3, 2]
    assert take(2, drop(2, count())) == [2, 3]

def test_first():
    if False:
        for i in range(10):
            print('nop')
    assert first('xyz') == 'x'
    assert first(count(7)) == 7
    assert first([]) is None

def test_second():
    if False:
        while True:
            i = 10
    assert second('xyz') == 'y'
    assert second(count(7)) == 8
    assert second('x') is None

def test_last():
    if False:
        for i in range(10):
            print('nop')
    assert last('xyz') == 'z'
    assert last(range(1, 10)) == 9
    assert last([]) is None
    assert last((x for x in 'xyz')) == 'z'

def test_nth():
    if False:
        while True:
            i = 10
    assert nth(0, 'xyz') == 'x'
    assert nth(2, 'xyz') == 'z'
    assert nth(3, 'xyz') is None
    assert nth(3, count(7)) == 10

def test_butlast():
    if False:
        while True:
            i = 10
    assert list(butlast('xyz')) == ['x', 'y']
    assert list(butlast([])) == []

def test_ilen():
    if False:
        while True:
            i = 10
    assert ilen('xyz') == 3
    assert ilen(range(10)) == 10

def test_lmap():
    if False:
        while True:
            i = 10
    assert lmap(_ * 2, [2, 3]) == [4, 6]
    assert lmap(None, [2, 3]) == [2, 3]
    assert lmap(_ + _, [1, 2], [4, 5]) == [5, 7]
    assert lmap('\\d+', ['a2', '13b']) == ['2', '13']
    assert lmap({'a': 1, 'b': 2}, 'ab') == [1, 2]
    assert lmap(set([1, 2, 3]), [0, 1, 2]) == [False, True, True]
    assert lmap(1, ['abc', '123']) == ['b', '2']
    assert lmap(slice(2), ['abc', '123']) == ['ab', '12']

def test_filter():
    if False:
        print('Hello World!')
    assert lfilter(None, [2, 3, 0]) == [2, 3]
    assert lfilter('\\d+', ['a2', '13b', 'c']) == ['a2', '13b']
    assert lfilter(set([1, 2, 3]), [0, 1, 2, 4, 1]) == [1, 2, 1]

def test_remove():
    if False:
        for i in range(10):
            print('nop')
    assert lremove(_ > 3, range(10)) == [0, 1, 2, 3]
    assert lremove('^a', ['a', 'b', 'ba']) == ['b', 'ba']

def test_keep():
    if False:
        return 10
    assert lkeep(_ % 3, range(5)) == [1, 2, 1]
    assert lkeep(range(5)) == [1, 2, 3, 4]
    assert lkeep(mapcat(range, range(4))) == [1, 1, 2]

def test_concat():
    if False:
        return 10
    assert lconcat('ab', 'cd') == list('abcd')
    assert lconcat() == []

def test_cat():
    if False:
        print('Hello World!')
    assert lcat('abcd') == list('abcd')
    assert lcat((range(x) for x in range(3))) == [0, 0, 1]

def test_flatten():
    if False:
        return 10
    assert lflatten([1, [2, 3]]) == [1, 2, 3]
    assert lflatten([[1, 2], 3]) == [1, 2, 3]
    assert lflatten([(2, 3)]) == [2, 3]
    assert lflatten([iter([2, 3])]) == [2, 3]

def test_flatten_follow():
    if False:
        while True:
            i = 10
    assert lflatten([1, [2, 3]], follow=is_list) == [1, 2, 3]
    assert lflatten([1, [(2, 3)]], follow=is_list) == [1, (2, 3)]

def test_mapcat():
    if False:
        for i in range(10):
            print('nop')
    assert lmapcat(lambda x: [x, x], 'abc') == list('aabbcc')

def test_interleave():
    if False:
        return 10
    assert list(interleave('ab', 'cd')) == list('acbd')
    assert list(interleave('ab_', 'cd')) == list('acbd')

def test_iterpose():
    if False:
        return 10
    assert list(interpose('.', 'abc')) == list('a.b.c')

def test_takewhile():
    if False:
        while True:
            i = 10
    assert list(takewhile([1, 2, None, 3])) == [1, 2]

def test_distinct():
    if False:
        i = 10
        return i + 15
    assert ldistinct('abcbad') == list('abcd')
    assert ldistinct([{}, {}, {'a': 1}, {'b': 2}], key=len) == [{}, {'a': 1}]
    assert ldistinct(['ab', 'cb', 'ad'], key=0) == ['ab', 'cb']

def test_split():
    if False:
        print('Hello World!')
    assert lmap(list, split(_ % 2, range(5))) == [[1, 3], [0, 2, 4]]

def test_lsplit():
    if False:
        while True:
            i = 10
    assert lsplit(_ % 2, range(5)) == ([1, 3], [0, 2, 4])
    with pytest.raises(TypeError):
        lsplit(2, range(5))

def test_split_at():
    if False:
        i = 10
        return i + 15
    assert lsplit_at(2, range(5)) == ([0, 1], [2, 3, 4])

def test_split_by():
    if False:
        print('Hello World!')
    assert lsplit_by(_ % 2, [1, 2, 3]) == ([1], [2, 3])

def test_group_by():
    if False:
        while True:
            i = 10
    assert group_by(_ % 2, range(5)) == {0: [0, 2, 4], 1: [1, 3]}
    assert group_by('\\d', ['a1', 'b2', 'c1']) == {'1': ['a1', 'c1'], '2': ['b2']}

def test_group_by_keys():
    if False:
        while True:
            i = 10
    assert group_by_keys('(\\d)(\\d)', ['12', '23']) == {'1': ['12'], '2': ['12', '23'], '3': ['23']}

def test_group_values():
    if False:
        for i in range(10):
            print('nop')
    assert group_values(['ab', 'ac', 'ba']) == {'a': ['b', 'c'], 'b': ['a']}

def test_count_by():
    if False:
        print('Hello World!')
    assert count_by(_ % 2, range(5)) == {0: 3, 1: 2}
    assert count_by('\\d', ['a1', 'b2', 'c1']) == {'1': 2, '2': 1}

def test_count_by_is_defaultdict():
    if False:
        for i in range(10):
            print('nop')
    cnts = count_by(len, [])
    assert cnts[1] == 0

def test_count_reps():
    if False:
        for i in range(10):
            print('nop')
    assert count_reps([0, 1, 0]) == {0: 2, 1: 1}

def test_partition():
    if False:
        i = 10
        return i + 15
    assert lpartition(2, [0, 1, 2, 3, 4]) == [[0, 1], [2, 3]]
    assert lpartition(2, 1, [0, 1, 2, 3]) == [[0, 1], [1, 2], [2, 3]]
    assert lpartition(2, iter(range(5))) == [[0, 1], [2, 3]]
    assert lmap(list, lpartition(2, range(5))) == [[0, 1], [2, 3]]

def test_chunks():
    if False:
        print('Hello World!')
    assert lchunks(2, [0, 1, 2, 3, 4]) == [[0, 1], [2, 3], [4]]
    assert lchunks(2, 1, [0, 1, 2, 3]) == [[0, 1], [1, 2], [2, 3], [3]]
    assert lchunks(3, 1, iter(range(3))) == [[0, 1, 2], [1, 2], [2]]

def test_partition_by():
    if False:
        print('Hello World!')
    assert lpartition_by(lambda x: x == 3, [1, 2, 3, 4, 5]) == [[1, 2], [3], [4, 5]]
    assert lpartition_by('x', 'abxcd') == [['a', 'b'], ['x'], ['c', 'd']]
    assert lpartition_by('\\d', '1211') == [['1'], ['2'], ['1', '1']]

def test_with_prev():
    if False:
        i = 10
        return i + 15
    assert list(with_prev(range(3))) == [(0, None), (1, 0), (2, 1)]

def test_with_next():
    if False:
        return 10
    assert list(with_next(range(3))) == [(0, 1), (1, 2), (2, None)]

def test_pairwise():
    if False:
        for i in range(10):
            print('nop')
    assert list(pairwise(range(3))) == [(0, 1), (1, 2)]

def test_lzip():
    if False:
        print('Hello World!')
    assert lzip('12', 'xy') == [('1', 'x'), ('2', 'y')]
    assert lzip('123', 'xy') == [('1', 'x'), ('2', 'y')]
    assert lzip('12', 'xyz') == [('1', 'x'), ('2', 'y')]
    assert lzip('12', iter('xyz')) == [('1', 'x'), ('2', 'y')]

def test_lzip_strict():
    if False:
        i = 10
        return i + 15
    assert lzip('123', 'xy', strict=False) == [('1', 'x'), ('2', 'y')]
    assert lzip('12', 'xy', strict=True) == [('1', 'x'), ('2', 'y')]
    assert lzip('12', iter('xy'), strict=True) == [('1', 'x'), ('2', 'y')]
    for wrap in (str, iter):
        with pytest.raises(ValueError):
            lzip(wrap('123'), wrap('xy'), strict=True)
        with pytest.raises(ValueError):
            lzip(wrap('12'), wrap('xyz'), wrap('abcd'), strict=True)
        with pytest.raises(ValueError):
            lzip(wrap('123'), wrap('xy'), wrap('abcd'), strict=True)
        with pytest.raises(ValueError):
            lzip(wrap('123'), wrap('xyz'), wrap('ab'), strict=True)

def test_reductions():
    if False:
        while True:
            i = 10
    assert lreductions(add, []) == []
    assert lreductions(add, [None]) == [None]
    assert lreductions(add, [1, 2, 3, 4]) == [1, 3, 6, 10]
    assert lreductions(lambda x, y: x + [y], [1, 2, 3], []) == [[1], [1, 2], [1, 2, 3]]

def test_sums():
    if False:
        while True:
            i = 10
    assert lsums([]) == []
    assert lsums([1, 2, 3, 4]) == [1, 3, 6, 10]
    assert lsums([[1], [2], [3]]) == [[1], [1, 2], [1, 2, 3]]

def test_without():
    if False:
        print('Hello World!')
    assert lwithout([]) == []
    assert lwithout([1, 2, 3, 4]) == [1, 2, 3, 4]
    assert lwithout([1, 2, 1, 0, 3, 1, 4], 0, 1) == [2, 3, 4]