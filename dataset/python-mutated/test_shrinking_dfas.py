import os
import sys
from contextlib import contextmanager
from itertools import islice
import pytest
from hypothesis import settings
from hypothesis.internal.conjecture.data import Status
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.internal.conjecture.shrinking import dfas
TEST_DFA_NAME = 'test name'

@contextmanager
def preserving_dfas():
    if False:
        return 10
    assert TEST_DFA_NAME not in dfas.SHRINKING_DFAS
    for k in dfas.SHRINKING_DFAS:
        assert not k.startswith(TEST_DFA_NAME)
    original = dict(dfas.SHRINKING_DFAS)
    try:
        yield
    finally:
        dfas.SHRINKING_DFAS.clear()
        dfas.SHRINKING_DFAS.update(original)
        dfas.update_learned_dfas()
    assert TEST_DFA_NAME not in dfas.SHRINKING_DFAS
    assert TEST_DFA_NAME not in dfas.learned_dfa_file.read_text(encoding='utf-8')

def test_updating_the_file_makes_no_changes_normally():
    if False:
        for i in range(10):
            print('nop')
    source1 = dfas.learned_dfa_file.read_text(encoding='utf-8')
    dfas.update_learned_dfas()
    source2 = dfas.learned_dfa_file.read_text(encoding='utf-8')
    assert source1 == source2

def test_updating_the_file_include_new_shrinkers():
    if False:
        i = 10
        return i + 15
    with preserving_dfas():
        source1 = dfas.learned_dfa_file.read_text(encoding='utf-8')
        dfas.SHRINKING_DFAS[TEST_DFA_NAME] = 'hello'
        dfas.update_learned_dfas()
        source2 = dfas.learned_dfa_file.read_text(encoding='utf-8')
        assert source1 != source2
        assert repr(TEST_DFA_NAME) in source2
    assert TEST_DFA_NAME not in dfas.SHRINKING_DFAS
    assert 'test name' not in dfas.learned_dfa_file.read_text(encoding='utf-8')

def called_by_shrinker():
    if False:
        print('Hello World!')
    frame = sys._getframe(0)
    while frame:
        fname = frame.f_globals.get('__file__', '')
        if os.path.basename(fname) == 'shrinker.py':
            return True
        frame = frame.f_back
    return False

def a_bad_test_function():
    if False:
        print('Hello World!')
    "Return a test function that we definitely can't normalize\n    because it cheats shamelessly and checks whether it's being\n    called by the shrinker and refuses to declare any new results\n    interesting."
    cache = {0: False}

    def test_function(data):
        if False:
            print('Hello World!')
        n = data.draw_bits(64)
        if n < 1000:
            return
        try:
            interesting = cache[n]
        except KeyError:
            interesting = cache.setdefault(n, not called_by_shrinker())
        if interesting:
            data.mark_interesting()
    return test_function

def test_will_error_if_does_not_normalise_and_cannot_update():
    if False:
        print('Hello World!')
    with pytest.raises(dfas.FailedToNormalise) as excinfo:
        dfas.normalize('bad', a_bad_test_function(), required_successes=10, allowed_to_update=False)
    assert 'not allowed' in excinfo.value.args[0]

def test_will_error_if_takes_too_long_to_normalize():
    if False:
        print('Hello World!')
    with preserving_dfas():
        with pytest.raises(dfas.FailedToNormalise) as excinfo:
            dfas.normalize('bad', a_bad_test_function(), required_successes=1000, allowed_to_update=True, max_dfas=0)
        assert 'too hard' in excinfo.value.args[0]

def non_normalized_test_function(data):
    if False:
        print('Hello World!')
    "This test function has two discrete regions that it\n    is hard to move between. It's basically unreasonable for\n    our shrinker to be able to transform from one to the other\n    because of how different they are."
    data.draw_bits(8)
    if data.draw_bits(1):
        n = data.draw_bits(10)
        if 100 < n < 1000:
            data.draw_bits(8)
            data.mark_interesting()
    else:
        n = data.draw_bits(64)
        if n > 10000:
            data.draw_bits(8)
            data.mark_interesting()

def test_can_learn_to_normalize_the_unnormalized():
    if False:
        return 10
    with preserving_dfas():
        prev = len(dfas.SHRINKING_DFAS)
        dfas.normalize(TEST_DFA_NAME, non_normalized_test_function, allowed_to_update=True)
        assert len(dfas.SHRINKING_DFAS) == prev + 1

def test_will_error_on_uninteresting_test():
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError):
        dfas.normalize(TEST_DFA_NAME, lambda data: data.draw_bits(64))

def test_makes_no_changes_if_already_normalized():
    if False:
        i = 10
        return i + 15

    def test_function(data):
        if False:
            i = 10
            return i + 15
        if data.draw_bits(16) >= 1000:
            data.mark_interesting()
    with preserving_dfas():
        before = dict(dfas.SHRINKING_DFAS)
        dfas.normalize(TEST_DFA_NAME, test_function, allowed_to_update=True)
        after = dict(dfas.SHRINKING_DFAS)
        assert after == before

def test_learns_to_bridge_only_two():
    if False:
        while True:
            i = 10

    def test_function(data):
        if False:
            return 10
        m = data.draw_bits(8)
        n = data.draw_bits(8)
        if (m, n) in ((10, 100), (2, 8)):
            data.mark_interesting()
    runner = ConjectureRunner(test_function, settings=settings(database=None), ignore_limits=True)
    dfa = dfas.learn_a_new_dfa(runner, [10, 100], [2, 8], lambda d: d.status == Status.INTERESTING)
    assert dfa.max_length(dfa.start) == 2
    assert list(map(list, dfa.all_matching_strings())) == [[2, 8], [10, 100]]

def test_learns_to_bridge_only_two_with_overlap():
    if False:
        for i in range(10):
            print('nop')
    u = [50, 0, 0, 0, 50]
    v = [50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50]

    def test_function(data):
        if False:
            print('Hello World!')
        for i in range(len(u)):
            c = data.draw_bits(8)
            if c != u[i]:
                if c != v[i]:
                    return
                break
        else:
            data.mark_interesting()
        for j in range(i + 1, len(v)):
            if data.draw_bits(8) != v[j]:
                return
        data.mark_interesting()
    runner = ConjectureRunner(test_function, settings=settings(database=None), ignore_limits=True)
    dfa = dfas.learn_a_new_dfa(runner, u, v, lambda d: d.status == Status.INTERESTING)
    assert list(islice(dfa.all_matching_strings(), 3)) == [b'', bytes(len(v) - len(u))]

def test_learns_to_bridge_only_two_with_suffix():
    if False:
        for i in range(10):
            print('nop')
    u = [7]
    v = [0] * 10 + [7]

    def test_function(data):
        if False:
            return 10
        n = data.draw_bits(8)
        if n == 7:
            data.mark_interesting()
        elif n != 0:
            return
        for _ in range(9):
            if data.draw_bits(8) != 0:
                return
        if data.draw_bits(8) == 7:
            data.mark_interesting()
    runner = ConjectureRunner(test_function, settings=settings(database=None), ignore_limits=True)
    dfa = dfas.learn_a_new_dfa(runner, u, v, lambda d: d.status == Status.INTERESTING)
    assert list(islice(dfa.all_matching_strings(), 3)) == [b'', bytes(len(v) - len(u))]