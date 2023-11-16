import pytest
import tests.plugin
from streamlink import Streamlink
from streamlink.api import streams

class TestStreamlinkAPI:

    @pytest.fixture(autouse=True)
    def _session(self, monkeypatch: pytest.MonkeyPatch, session: Streamlink):
        if False:
            return 10
        monkeypatch.setattr('streamlink.api.Streamlink', lambda : session)
        session.load_plugins(tests.plugin.__path__[0])

    def test_find_test_plugin(self):
        if False:
            while True:
                i = 10
        assert 'hls' in streams('test.se')

    def test_no_streams_exception(self):
        if False:
            for i in range(10):
                print('nop')
        assert streams('test.se/NoStreamsError') == {}

    def test_no_streams(self):
        if False:
            for i in range(10):
                print('nop')
        assert streams('test.se/empty') == {}

    def test_stream_type_filter(self):
        if False:
            return 10
        stream_types = ['hls']
        available_streams = streams('test.se', stream_types=stream_types)
        assert 'hls' in available_streams
        assert 'test' not in available_streams
        assert 'http' not in available_streams

    def test_stream_type_wildcard(self):
        if False:
            return 10
        stream_types = ['hls', '*']
        available_streams = streams('test.se', stream_types=stream_types)
        assert 'hls' in available_streams
        assert 'test' in available_streams
        assert 'http' in available_streams