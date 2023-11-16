import pytest
from qutebrowser.config import websettings
from qutebrowser.misc import objects
from qutebrowser.utils import usertypes

@pytest.mark.parametrize(['user_agent', 'os_info', 'webkit_version', 'upstream_browser_key', 'upstream_browser_version', 'qt_key'], [('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) QtWebEngine/5.14.0 Chrome/77.0.3865.98 Safari/537.36', 'X11; Linux x86_64', '537.36', 'Chrome', '77.0.3865.98', 'QtWebEngine'), ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/602.1 (KHTML, like Gecko) qutebrowser/1.8.3 Version/10.0 Safari/602.1', 'X11; Linux x86_64', '602.1', 'Version', '10.0', 'Qt'), ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) QtWebEngine/5.13.2 Chrome/73.0.3683.105 Safari/537.36', 'Macintosh; Intel Mac OS X 10_12_6', '537.36', 'Chrome', '73.0.3683.105', 'QtWebEngine'), ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) QtWebEngine/5.12.5 Chrome/69.0.3497.128 Safari/537.36', 'Windows NT 10.0; Win64; x64', '537.36', 'Chrome', '69.0.3497.128', 'QtWebEngine')])
def test_parse_user_agent(user_agent, os_info, webkit_version, upstream_browser_key, upstream_browser_version, qt_key):
    if False:
        for i in range(10):
            print('nop')
    parsed = websettings.UserAgent.parse(user_agent)
    assert parsed.os_info == os_info
    assert parsed.webkit_version == webkit_version
    assert parsed.upstream_browser_key == upstream_browser_key
    assert parsed.upstream_browser_version == upstream_browser_version
    assert parsed.qt_key == qt_key

def test_user_agent(monkeypatch, config_stub, qapp):
    if False:
        while True:
            i = 10
    webenginesettings = pytest.importorskip('qutebrowser.browser.webengine.webenginesettings')
    monkeypatch.setattr(objects, 'backend', usertypes.Backend.QtWebEngine)
    webenginesettings.init_user_agent()
    config_stub.val.content.headers.user_agent = 'test {qt_key}'
    assert websettings.user_agent() == 'test QtWebEngine'
    config_stub.val.content.headers.user_agent = 'test2 {qt_key}'
    assert websettings.user_agent() == 'test2 QtWebEngine'

def test_config_init(request, monkeypatch, config_stub):
    if False:
        print('Hello World!')
    if request.config.webengine:
        from qutebrowser.browser.webengine import webenginesettings
        monkeypatch.setattr(webenginesettings, 'init', lambda : None)
    else:
        from qutebrowser.browser.webkit import webkitsettings
        monkeypatch.setattr(webkitsettings, 'init', lambda : None)
    websettings.init(args=None)
    assert config_stub.dump_userconfig() == '<Default configuration>'