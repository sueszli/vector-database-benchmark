import requests_mock
from streamlink import Streamlink
from streamlink.plugins.rtpplay import RTPPlay
from streamlink.stream.hls import HLSStream
from tests.plugins import PluginCanHandleUrl
from tests.resources import text

class TestPluginCanHandleUrlRTPPlay(PluginCanHandleUrl):
    __plugin__ = RTPPlay
    should_match = ['http://www.rtp.pt/play/', 'https://www.rtp.pt/play/', 'https://www.rtp.pt/play/direto/rtp1', 'https://www.rtp.pt/play/direto/rtpmadeira']
    should_not_match = ['https://www.rtp.pt/programa/', 'http://www.rtp.pt/programa/', 'https://media.rtp.pt/', 'http://media.rtp.pt/']

class TestRTPPlay:
    _content_pre = '\n        /*  var player1 = new RTPPlayer({\n                file: {hls : atob( decodeURIComponent(["aHR0c", "HM6Ly", "9pbnZ", "hbGlk"].join("") ) ), dash : foo() } }); */\n        var f = {hls : atob( decodeURIComponent(["aHR0c", "HM6Ly", "9pbnZ", "hbGlk"].join("") ) ), dash: foo() };\n    '
    _content_invalid = '\n        var f = {hls : ""};\n    '
    _content_valid = '\n        var f = {hls : decodeURIComponent(["https%3","A%2F%2F", "valid"].join("") ), dash: foo() };\n    '
    _content_valid_b64 = '\n        var f = {hls : atob( decodeURIComponent(["aHR0c", "HM6Ly", "92YWx", "pZA=="].join("") ) ), dash: foo() };\n    '

    @property
    def playlist(self):
        if False:
            for i in range(10):
                print('nop')
        with text('hls/test_master.m3u8') as pl:
            return pl.read()

    def subject(self, url, response):
        if False:
            return 10
        with requests_mock.Mocker() as mock:
            mock.get(url=url, text=response)
            mock.get('https://valid', text=self.playlist)
            mock.get('https://invalid', exc=AssertionError)
            session = Streamlink()
            plugin = RTPPlay(session, url)
            return plugin._get_streams()

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        streams = self.subject('https://www.rtp.pt/play/id/title', '')
        assert streams is None

    def test_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        streams = self.subject('https://www.rtp.pt/play/id/title', self._content_pre + self._content_invalid)
        assert streams is None

    def test_valid(self):
        if False:
            print('Hello World!')
        streams = self.subject('https://www.rtp.pt/play/id/title', self._content_pre + self._content_valid)
        assert isinstance(next(iter(streams.values())), HLSStream)

    def test_valid_b64(self):
        if False:
            i = 10
            return i + 15
        streams = self.subject('https://www.rtp.pt/play/id/title', self._content_pre + self._content_valid_b64)
        assert isinstance(next(iter(streams.values())), HLSStream)