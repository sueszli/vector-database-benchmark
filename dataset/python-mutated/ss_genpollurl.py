import pytest
import libqtile.widget
from test.widgets.test_generic_poll_text import MockRequest, Mockurlopen

@pytest.fixture
def widget(monkeypatch):
    if False:
        i = 10
        return i + 15
    MockRequest.return_value = b'Text from URL'
    monkeypatch.setattr('libqtile.widget.generic_poll_text.Request', MockRequest)
    monkeypatch.setattr('libqtile.widget.generic_poll_text.urlopen', Mockurlopen)
    yield libqtile.widget.GenPollUrl

@pytest.mark.parametrize('screenshot_manager', [{}, {'url': 'http://test.qtile.org', 'json': False, 'parse': lambda x: x}], indirect=True)
def ss_genpollurl(screenshot_manager):
    if False:
        print('Hello World!')
    screenshot_manager.take_screenshot()