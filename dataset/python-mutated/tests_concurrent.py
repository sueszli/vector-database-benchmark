"""
Tests for `tqdm.contrib.concurrent`.
"""
from pytest import warns
from tqdm.contrib.concurrent import process_map, thread_map
from .tests_tqdm import StringIO, TqdmWarning, closing, importorskip, mark, skip

def incr(x):
    if False:
        for i in range(10):
            print('nop')
    'Dummy function'
    return x + 1

def test_thread_map():
    if False:
        i = 10
        return i + 15
    'Test contrib.concurrent.thread_map'
    with closing(StringIO()) as our_file:
        a = range(9)
        b = [i + 1 for i in a]
        try:
            assert thread_map(lambda x: x + 1, a, file=our_file) == b
        except ImportError as err:
            skip(str(err))
        assert thread_map(incr, a, file=our_file) == b

def test_process_map():
    if False:
        while True:
            i = 10
    'Test contrib.concurrent.process_map'
    with closing(StringIO()) as our_file:
        a = range(9)
        b = [i + 1 for i in a]
        try:
            assert process_map(incr, a, file=our_file) == b
        except ImportError as err:
            skip(str(err))

@mark.parametrize('iterables,should_warn', [([], False), (['x'], False), ([()], False), (['x', ()], False), (['x' * 1001], True), (['x' * 100, ('x',) * 1001], True)])
def test_chunksize_warning(iterables, should_warn):
    if False:
        i = 10
        return i + 15
    'Test contrib.concurrent.process_map chunksize warnings'
    patch = importorskip('unittest.mock').patch
    with patch('tqdm.contrib.concurrent._executor_map'):
        if should_warn:
            warns(TqdmWarning, process_map, incr, *iterables)
        else:
            process_map(incr, *iterables)