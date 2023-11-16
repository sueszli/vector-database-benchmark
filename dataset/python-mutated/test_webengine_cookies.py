import pytest
from qutebrowser.qt.core import QUrl
pytest.importorskip('qutebrowser.qt.webenginecore')
from qutebrowser.qt.webenginecore import QWebEngineCookieStore, QWebEngineProfile
from qutebrowser.browser.webengine import cookies
from qutebrowser.utils import urlmatch

@pytest.fixture
def filter_request():
    if False:
        return 10
    request = QWebEngineCookieStore.FilterRequest()
    request.firstPartyUrl = QUrl('https://example.com')
    return request

@pytest.fixture(autouse=True)
def enable_cookie_logging(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr(cookies.objects, 'debug_flags', ['log-cookies'])

@pytest.mark.parametrize('setting, third_party, accepted', [('all', False, True), ('never', False, False), ('no-3rdparty', False, True), ('no-3rdparty', True, False)])
def test_accept_cookie(config_stub, filter_request, setting, third_party, accepted):
    if False:
        i = 10
        return i + 15
    'Test that _accept_cookie respects content.cookies.accept.'
    config_stub.val.content.cookies.accept = setting
    filter_request.thirdParty = third_party
    assert cookies._accept_cookie(filter_request) == accepted

@pytest.mark.parametrize('setting, pattern_setting, third_party, accepted', [('never', 'all', False, True), ('all', 'never', False, False), ('no-3rdparty', 'all', True, True), ('all', 'no-3rdparty', True, False)])
def test_accept_cookie_with_pattern(config_stub, filter_request, setting, pattern_setting, third_party, accepted):
    if False:
        print('Hello World!')
    'Test that _accept_cookie matches firstPartyUrl with the UrlPattern.'
    filter_request.thirdParty = third_party
    config_stub.set_str('content.cookies.accept', setting)
    config_stub.set_str('content.cookies.accept', pattern_setting, pattern=urlmatch.UrlPattern('https://*.example.com'))
    assert cookies._accept_cookie(filter_request) == accepted

@pytest.mark.parametrize('global_value', ['never', 'all'])
def test_invalid_url(config_stub, filter_request, global_value):
    if False:
        while True:
            i = 10
    "Make sure we fall back to the global value with invalid URLs.\n\n    This can happen when there's a cookie request from an iframe, e.g. here:\n    https://developers.google.com/youtube/youtube_player_demo\n    "
    config_stub.val.content.cookies.accept = global_value
    filter_request.firstPartyUrl = QUrl()
    accepted = global_value == 'all'
    assert cookies._accept_cookie(filter_request) == accepted

@pytest.mark.parametrize('enabled', [True, False])
def test_logging(monkeypatch, config_stub, filter_request, caplog, enabled):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(cookies.objects, 'debug_flags', ['log-cookies'] if enabled else [])
    config_stub.val.content.cookies.accept = 'all'
    caplog.clear()
    cookies._accept_cookie(filter_request)
    if enabled:
        expected = 'Cookie from origin <unknown> on https://example.com (third party: False) -> applying setting all'
        assert caplog.messages == [expected]
    else:
        assert not caplog.messages

class TestInstall:

    def test_real_profile(self):
        if False:
            i = 10
            return i + 15
        profile = QWebEngineProfile()
        cookies.install_filter(profile)

    def test_fake_profile(self, stubs):
        if False:
            while True:
                i = 10
        store = stubs.FakeCookieStore()
        profile = stubs.FakeWebEngineProfile(cookie_store=store)
        cookies.install_filter(profile)
        assert store.cookie_filter is cookies._accept_cookie