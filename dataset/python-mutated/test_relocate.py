from __future__ import annotations
import pytest
import ibis
import ibis.common.exceptions as exc
import ibis.selectors as s

def test_individual_columns():
    if False:
        i = 10
        return i + 15
    t = ibis.table(dict(x='int', y='int'))
    assert t.relocate('x', after='y').columns == list('yx')
    assert t.relocate('y', before='x').columns == list('yx')

def test_move_blocks():
    if False:
        print('Hello World!')
    t = ibis.table(dict(x='int', a='string', y='int', b='string'))
    assert t.relocate(s.of_type('string')).columns == list('abxy')
    assert t.relocate(s.of_type('string'), after=s.numeric()).columns == list('xyab')

def test_duplicates_not_renamed():
    if False:
        print('Hello World!')
    t = ibis.table(dict(x='int', y='int'))
    assert t.relocate('y', s.numeric()).columns == list('yx')
    assert t.relocate('y', s.numeric(), 'y').columns == list('yx')

def test_keep_non_contiguous_variables():
    if False:
        i = 10
        return i + 15
    t = ibis.table(dict.fromkeys('abcde', 'int'))
    assert t.relocate('b', after=s.c('a', 'c', 'e')).columns == list('acdeb')
    assert t.relocate('e', before=s.c('b', 'd')).columns == list('aebcd')

def test_before_after_does_not_move_to_front():
    if False:
        print('Hello World!')
    t = ibis.table(dict(x='int', y='int'))
    assert t.relocate('y').columns == list('yx')

def test_only_one_of_before_and_after():
    if False:
        for i in range(10):
            print('nop')
    t = ibis.table(dict(x='int', y='int', z='int'))
    with pytest.raises(exc.IbisInputError, match='Cannot specify both'):
        t.relocate('z', before='x', after='y')

def test_respects_order():
    if False:
        return 10
    t = ibis.table(dict.fromkeys('axbzy', 'int'))
    assert t.relocate('x', 'y', 'z', before='x').columns == list('axyzb')
    assert t.relocate('x', 'y', 'z', before=s.last()).columns == list('abxyz')
    assert t.relocate('x', 'a', 'z').columns == list('xazby')

def test_relocate_can_rename():
    if False:
        print('Hello World!')
    t = ibis.table(dict(a='int', b='int', c='int', d='string', e='string', f='string'))
    assert t.relocate(ffff='f').columns == ['ffff', *'abcde']
    assert t.relocate(ffff='f', before='c').columns == [*'ab', 'ffff', *'cde']
    assert t.relocate(ffff='f', after='c').columns == [*'abc', 'ffff', *'de']

def test_retains_last_duplicate_when_renaming_and_moving():
    if False:
        while True:
            i = 10
    t = ibis.table(dict(x='int'))
    assert t.relocate(a='x', b='x').columns == ['b']
    t = ibis.table(dict(x='int', y='int'))
    assert t.relocate(a='x', b='y', c='x').columns == list('bc')

def test_everything():
    if False:
        for i in range(10):
            print('nop')
    t = ibis.table(dict(w='int', x='int', y='int', z='int'))
    assert t.relocate('y', 'z', before=s.all()).columns == list('yzwx')
    assert t.relocate('y', 'z', after=s.all()).columns == list('wxyz')

def test_moves_to_front_with_no_before_and_no_after():
    if False:
        for i in range(10):
            print('nop')
    t = ibis.table(dict(x='int', y='int', z='int'))
    assert t.relocate('z', 'y').columns == list('zyx')

def test_empty_before_moves_to_front():
    if False:
        while True:
            i = 10
    t = ibis.table(dict(x='int', y='int', z='int'))
    assert t.relocate('y', before=s.of_type('string')).columns == list('yxz')

def test_empty_after_moves_to_end():
    if False:
        for i in range(10):
            print('nop')
    t = ibis.table(dict(x='int', y='int', z='int'))
    assert t.relocate('y', after=s.of_type('string')).columns == list('xzy')

def test_no_arguments():
    if False:
        for i in range(10):
            print('nop')
    t = ibis.table(dict(x='int', y='int', z='int'))
    with pytest.raises(exc.IbisInputError, match='At least one selector'):
        assert t.relocate()