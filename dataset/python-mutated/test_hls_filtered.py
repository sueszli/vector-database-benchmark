import unittest
from threading import Event
from unittest.mock import MagicMock, call, patch
import pytest
from streamlink.stream.hls import HLSStream, HLSStreamReader
from tests.mixins.stream_hls import EventedHLSStreamWriter, Playlist, Segment, TestMixinStreamHLS
FILTERED = 'filtered'
TIMEOUT_HANDSHAKE = 5

class SegmentFiltered(Segment):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.title = FILTERED

class _TestSubjectHLSReader(HLSStreamReader):
    __writer__ = EventedHLSStreamWriter

class _TestSubjectHLSStream(HLSStream):
    __reader__ = _TestSubjectHLSReader

@patch('streamlink.stream.hls.HLSStreamWorker.wait', MagicMock(return_value=True))
class TestFilteredHLSStream(TestMixinStreamHLS, unittest.TestCase):
    __stream__ = _TestSubjectHLSStream

    @classmethod
    def filter_segment(cls, sequence):
        if False:
            while True:
                i = 10
        return sequence.title == FILTERED

    def get_session(self, options=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        session = super().get_session(options)
        session.set_option('hls-live-edge', 2)
        session.set_option('stream-timeout', 0)
        return session

    def test_not_filtered(self):
        if False:
            i = 10
            return i + 15
        segments = self.subject([Playlist(0, [SegmentFiltered(0), SegmentFiltered(1)], end=True)])
        self.await_write(2)
        data = self.await_read()
        assert data == self.content(segments), 'Does not filter by default'
        assert self.thread.reader.filter_wait(timeout=0)

    @patch('streamlink.stream.hls.HLSStreamWriter.should_filter_segment', new=filter_segment)
    @patch('streamlink.stream.hls.hls.log')
    def test_filtered_logging(self, mock_log):
        if False:
            return 10
        segments = self.subject([Playlist(0, [SegmentFiltered(0), SegmentFiltered(1)]), Playlist(2, [Segment(2), Segment(3)]), Playlist(4, [SegmentFiltered(4), SegmentFiltered(5)]), Playlist(6, [Segment(6), Segment(7)], end=True)])
        data = b''
        assert not self.thread.reader.is_paused(), "Doesn't let the reader wait if not filtering"
        self.await_write(2)
        assert mock_log.info.call_args_list == [call('Filtering out segments and pausing stream output')]
        assert mock_log.warning.call_args_list == [], "Doesn't warn about discontinuities when filtering pre-rolls"
        assert self.thread.reader.is_paused(), 'Lets the reader wait if filtering'
        self.await_write(2)
        assert mock_log.info.call_args_list == [call('Filtering out segments and pausing stream output'), call('Resuming stream output')]
        assert mock_log.warning.call_args_list == [], "Doesn't warn about discontinuities when resuming after pre-rolls"
        assert not self.thread.reader.is_paused(), "Doesn't let the reader wait if not filtering"
        data += self.await_read()
        self.await_write(2)
        assert mock_log.info.call_args_list == [call('Filtering out segments and pausing stream output'), call('Resuming stream output'), call('Filtering out segments and pausing stream output')]
        assert mock_log.warning.call_args_list == [], "Doesn't warn about discontinuities when filtering mid-rolls"
        assert self.thread.reader.is_paused(), 'Lets the reader wait if filtering'
        self.await_write(2)
        assert mock_log.info.call_args_list == [call('Filtering out segments and pausing stream output'), call('Resuming stream output'), call('Filtering out segments and pausing stream output'), call('Resuming stream output')]
        assert mock_log.warning.call_args_list == [call('Encountered a stream discontinuity. This is unsupported and will result in incoherent output data.')], 'Warns about discontinuities when resuming after mid-rolls'
        assert not self.thread.reader.is_paused(), "Doesn't let the reader wait if not filtering"
        data += self.await_read()
        assert data == self.content(segments, cond=lambda s: s.num % 4 > 1), 'Correctly filters out segments'
        assert all((self.called(s) for s in segments.values())), 'Downloads all segments'

    @patch('streamlink.stream.hls.HLSStreamWriter.should_filter_segment', new=filter_segment)
    def test_filtered_timeout(self):
        if False:
            return 10
        segments = self.subject([Playlist(0, [Segment(0), Segment(1)], end=True)])
        self.await_write()
        data = self.await_read()
        assert data == segments[0].content, 'Has read the first segment'
        with pytest.raises(OSError, match='^Read timeout$'):
            self.await_read()

    @patch('streamlink.stream.hls.HLSStreamWriter.should_filter_segment', new=filter_segment)
    def test_filtered_no_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        segments = self.subject([Playlist(0, [SegmentFiltered(0), SegmentFiltered(1)]), Playlist(2, [Segment(2), Segment(3)], end=True)])
        assert not self.thread.reader.is_paused(), "Doesn't let the reader wait if not filtering"
        self.await_write(2)
        assert self.thread.reader.is_paused(), 'Lets the reader wait if filtering'
        assert not self.thread.reader.filter_wait(timeout=0), 'Is filtering'
        self.thread.handshake.go()
        self.await_write()
        assert not self.thread.reader.is_paused(), 'Reader is not waiting anymore'
        assert self.thread.handshake.wait_done(TIMEOUT_HANDSHAKE), "Doesn't time out when filtering"
        assert b''.join(self.thread.data) == segments[2].content, 'Reads next available buffer data'
        self.await_write()
        data = self.await_read()
        assert data == self.content(segments, cond=lambda s: s.num >= 2)

    @patch('streamlink.stream.hls.HLSStreamWriter.should_filter_segment', new=filter_segment)
    def test_filtered_closed(self):
        if False:
            for i in range(10):
                print('nop')
        self.subject(start=False, playlists=[Playlist(0, [SegmentFiltered(0), SegmentFiltered(1)], end=True)])
        event_filter_wait_called = Event()
        orig_wait = self.thread.reader._event_filter.wait

        def mocked_wait(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            event_filter_wait_called.set()
            return orig_wait(*args, **kwargs)
        with patch.object(self.thread.reader._event_filter, 'wait', side_effect=mocked_wait):
            self.start()
            assert not self.thread.reader.is_paused(), "Doesn't let the reader wait if not filtering"
            self.await_write()
            assert self.thread.reader.is_paused(), 'Lets the reader wait if filtering'
            self.thread.handshake.go()
            assert event_filter_wait_called.wait(TIMEOUT_HANDSHAKE), 'Missing event_filter.wait() call'
            self.thread.reader.close()
            assert self.thread.handshake.wait_done(TIMEOUT_HANDSHAKE), 'Is not a read timeout on stream close'
            assert self.thread.data == [b''], 'Stops reading on stream close'

    def test_hls_segment_ignore_names(self):
        if False:
            while True:
                i = 10
        segments = self.subject([Playlist(0, [Segment(0), Segment(1), Segment(2), Segment(3)], end=True)], {'hls-segment-ignore-names': ['.*', 'segment0', 'segment2']})
        self.await_write(4)
        data = self.await_read()
        assert data == self.content(segments, cond=lambda s: s.num % 2 > 0)