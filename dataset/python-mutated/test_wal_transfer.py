import gevent
import pytest
from fast_wait import fast_wait
from wal_e import worker
from wal_e.exception import UserCritical
assert fast_wait

class Explosion(Exception):
    """Marker type for fault injection."""
    pass

class FakeWalSegment(object):

    def __init__(self, seg_path, explicit=False, upload_explosive=False, mark_done_explosive=False):
        if False:
            print('Hello World!')
        self.explicit = explicit
        self._upload_explosive = upload_explosive
        self._mark_done_explosive = mark_done_explosive
        self._marked = False
        self._uploaded = False

    def mark_done(self):
        if False:
            while True:
                i = 10
        if self._mark_done_explosive:
            raise self._mark_done_explosive
        self._marked = True

class FakeWalUploader(object):

    def __call__(self, segment):
        if False:
            print('Hello World!')
        if segment._upload_explosive:
            raise segment._upload_explosive
        segment._uploaded = True
        return segment

def failed(seg):
    if False:
        for i in range(10):
            print('nop')
    'Returns true if a segment could be a failed upload.\n\n    Or in progress, the two are not distinguished.\n    '
    return seg._marked is False and seg._uploaded is False

def success(seg):
    if False:
        for i in range(10):
            print('nop')
    "Returns true if a segment has been successfully uploaded.\n\n    Checks that mark_done was not called if this is an 'explicit' wal\n    segment from Postgres.\n    "
    if seg.explicit:
        assert seg._marked is False
    return seg._uploaded

def indeterminate(seg):
    if False:
        print('Hello World!')
    'Returns true as long as the segment is internally consistent.\n\n    Checks invariants of mark_done, depending on whether the segment\n    has been uploaded.  This is useful in cases with tests with\n    failures and concurrent execution, and calls out the state of the\n    segment in any case to the reader.\n    '
    if seg._uploaded:
        if seg.explicit:
            assert seg._marked is False
        else:
            assert seg._marked is True
    else:
        assert seg._marked is False
    return True

def prepare_multi_upload_segments():
    if False:
        i = 10
        return i + 15
    'Prepare a handful of fake segments for upload.'
    yield FakeWalSegment('0' * 8 * 3, explicit=True)
    for i in range(1, 5):
        yield FakeWalSegment(str(i) * 8 * 3, explicit=False)

def test_simple_upload():
    if False:
        while True:
            i = 10
    'Model a case where there is no concurrency while uploading.'
    group = worker.WalTransferGroup(FakeWalUploader())
    seg = FakeWalSegment('1' * 8 * 3, explicit=True)
    group.start(seg)
    group.join()
    assert success(seg)

def test_multi_upload():
    if False:
        print('Hello World!')
    'Model a case with upload concurrency.'
    group = worker.WalTransferGroup(FakeWalUploader())
    segments = list(prepare_multi_upload_segments())
    for seg in segments:
        group.start(seg)
    group.join()
    for seg in segments:
        assert success(seg)

def test_simple_fail():
    if False:
        return 10
    'Model a simple failure in the non-concurrent case.'
    group = worker.WalTransferGroup(FakeWalUploader())
    exp = Explosion('fail')
    seg = FakeWalSegment('1' * 8 * 3, explicit=True, upload_explosive=exp)
    group.start(seg)
    with pytest.raises(Explosion) as e:
        group.join()
    assert e.value is exp
    assert failed(seg)

def test_multi_explicit_fail():
    if False:
        return 10
    'Model a failure of the explicit segment under concurrency.'
    group = worker.WalTransferGroup(FakeWalUploader())
    segments = list(prepare_multi_upload_segments())
    exp = Explosion('fail')
    segments[0]._upload_explosive = exp
    for seg in segments:
        group.start(seg)
    with pytest.raises(Explosion) as e:
        group.join()
    assert e.value is exp
    assert failed(segments[0])
    for seg in segments[1:]:
        assert success(seg)

def test_multi_pipeline_fail():
    if False:
        while True:
            i = 10
    'Model a failure of the pipelined segments under concurrency.'
    group = worker.WalTransferGroup(FakeWalUploader())
    segments = list(prepare_multi_upload_segments())
    exp = Explosion('fail')
    fail_idx = 2
    segments[fail_idx]._upload_explosive = exp
    for seg in segments:
        group.start(seg)
    with pytest.raises(Explosion) as e:
        group.join()
    assert e.value is exp
    for (i, seg) in enumerate(segments):
        if i == fail_idx:
            assert failed(seg)
        else:
            assert indeterminate(seg)

def test_finally_execution():
    if False:
        i = 10
        return i + 15
    'When one segment fails ensure parallel segments clean up.'
    segBad = FakeWalSegment('1' * 8 * 3)
    segOK = FakeWalSegment('2' * 8 * 3)

    class CleanupCheckingUploader(object):

        def __init__(self):
            if False:
                print('Hello World!')
            self.cleaned_up = False

        def __call__(self, segment):
            if False:
                i = 10
                return i + 15
            if segment is segOK:
                try:
                    while True:
                        gevent.sleep(0.1)
                finally:
                    self.cleaned_up = True
            elif segment is segBad:
                raise Explosion('fail')
            else:
                assert False, 'Expect only two segments'
            segment._uploaded = True
            return segment
    uploader = CleanupCheckingUploader()
    group = worker.WalTransferGroup(uploader)
    group.start(segOK)
    group.start(segBad)
    with pytest.raises(Explosion):
        group.join()
    assert uploader.cleaned_up is True

def test_start_after_join():
    if False:
        for i in range(10):
            print('nop')
    'Break an invariant by adding transfers after .join.'
    group = worker.WalTransferGroup(FakeWalUploader())
    group.join()
    seg = FakeWalSegment('arbitrary')
    with pytest.raises(UserCritical):
        group.start(seg)

def test_mark_done_fault():
    if False:
        i = 10
        return i + 15
    'Exercise exception handling from .mark_done()'
    group = worker.WalTransferGroup(FakeWalUploader())
    exp = Explosion('boom')
    seg = FakeWalSegment('arbitrary', mark_done_explosive=exp)
    group.start(seg)
    with pytest.raises(Explosion) as e:
        group.join()
    assert e.value is exp