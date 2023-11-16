import pytest
from stage_pgxlog import pg_xlog
from wal_e import worker
from wal_e import exception
assert pg_xlog

def make_segment(num, **kwargs):
    if False:
        i = 10
        return i + 15
    return worker.WalSegment('pg_xlog/' + str(num) * 8 * 3, **kwargs)

def test_simple_create():
    if False:
        i = 10
        return i + 15
    'Check __init__.'
    make_segment(1)

def test_mark_done_invariant():
    if False:
        for i in range(10):
            print('nop')
    "Check explicit segments cannot be .mark_done'd."
    seg = make_segment(1, explicit=True)
    with pytest.raises(exception.UserCritical):
        seg.mark_done()

def test_mark_done(pg_xlog):
    if False:
        i = 10
        return i + 15
    "Check non-explicit segments can be .mark_done'd."
    seg = make_segment(1, explicit=False)
    pg_xlog.touch(seg.name, '.ready')
    seg.mark_done()

def test_mark_done_problem(pg_xlog, monkeypatch):
    if False:
        print('Hello World!')
    'Check that mark_done fails loudly if status file is missing.\n\n    While in normal operation, WAL-E does not expect races against\n    other processes manipulating .ready files.  But, just in case that\n    should occur, WAL-E is designed to crash, exercised here.\n    '
    seg = make_segment(1, explicit=False)
    with pytest.raises(exception.UserCritical):
        seg.mark_done()

def test_simple_search(pg_xlog):
    if False:
        while True:
            i = 10
    'Must find a .ready file'
    name = '1' * 8 * 3
    pg_xlog.touch(name, '.ready')
    segs = worker.WalSegment.from_ready_archive_status('pg_xlog')
    assert next(segs).path == 'pg_xlog/' + name
    with pytest.raises(StopIteration):
        next(segs)

def test_multi_search(pg_xlog):
    if False:
        return 10
    'Test finding a few ready files.\n\n    Also throw in some random junk to make sure they are filtered out\n    from processing correctly.\n    '
    for i in range(3):
        ready = str(i) * 8 * 3
        pg_xlog.touch(ready, '.ready')
    complete_segment_name = 'F' * 8 * 3
    pg_xlog.touch(complete_segment_name, '.done')
    ready_history_file_name = 'F' * 8 + '.history'
    pg_xlog.touch(ready_history_file_name, '.ready')
    segs = worker.WalSegment.from_ready_archive_status(str(pg_xlog.pg_xlog))
    for (i, seg) in enumerate(segs):
        assert seg.name == str(i) * 8 * 3
    assert i == 2
    pg_xlog.assert_exists(complete_segment_name, '.done')
    pg_xlog.assert_exists(ready_history_file_name, '.ready')