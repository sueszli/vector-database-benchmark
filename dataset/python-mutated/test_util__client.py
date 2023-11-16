from __future__ import annotations
import pytest
pytest
import bokeh.client.util as bcu

class Test_server_url_for_websocket_url:

    def test_with_ws(self) -> None:
        if False:
            return 10
        assert bcu.server_url_for_websocket_url('ws://foo.com/ws') == 'http://foo.com/'

    def test_with_wss(self) -> None:
        if False:
            while True:
                i = 10
        assert bcu.server_url_for_websocket_url('wss://foo.com/ws') == 'https://foo.com/'

    def test_bad_proto(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            bcu.server_url_for_websocket_url('junk://foo.com/ws')

    def test_bad_ending(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            bcu.server_url_for_websocket_url('ws://foo.com/junk')
        with pytest.raises(ValueError):
            bcu.server_url_for_websocket_url('wss://foo.com/junk')

class Test_websocket_url_for_server_url:

    def test_with_http(self) -> None:
        if False:
            return 10
        assert bcu.websocket_url_for_server_url('http://foo.com') == 'ws://foo.com/ws'
        assert bcu.websocket_url_for_server_url('http://foo.com/') == 'ws://foo.com/ws'

    def test_with_https(self) -> None:
        if False:
            print('Hello World!')
        assert bcu.websocket_url_for_server_url('https://foo.com') == 'wss://foo.com/ws'
        assert bcu.websocket_url_for_server_url('https://foo.com/') == 'wss://foo.com/ws'

    def test_bad_proto(self) -> None:
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            bcu.websocket_url_for_server_url('junk://foo.com')