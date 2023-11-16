"""Check how Qt behaves when trying to execute JS."""
import pytest

@pytest.mark.parametrize('js_enabled, expected', [(True, 2.0), (False, None)])
def test_simple_js_webkit(webview, js_enabled, expected):
    if False:
        i = 10
        return i + 15
    'With QtWebKit, evaluateJavaScript works when JS is on.'
    from qutebrowser.qt.webkit import QWebSettings
    webview.settings().setAttribute(QWebSettings.WebAttribute.JavascriptEnabled, js_enabled)
    result = webview.page().mainFrame().evaluateJavaScript('1 + 1')
    assert result == expected

@pytest.mark.parametrize('js_enabled, expected', [(True, 2.0), (False, 2.0)])
def test_element_js_webkit(webview, js_enabled, expected):
    if False:
        for i in range(10):
            print('nop')
    'With QtWebKit, evaluateJavaScript on an element works with JS off.'
    from qutebrowser.qt.webkit import QWebSettings
    webview.settings().setAttribute(QWebSettings.WebAttribute.JavascriptEnabled, js_enabled)
    elem = webview.page().mainFrame().documentElement()
    result = elem.evaluateJavaScript('1 + 1')
    assert result == expected

@pytest.mark.usefixtures('redirect_webengine_data')
@pytest.mark.parametrize('js_enabled, world, expected', [(True, 0, 2.0), (False, 0, None), (True, 1, 2.0), (False, 1, 2.0), (True, 2, 2.0), (False, 2, 2.0)])
def test_simple_js_webengine(qtbot, webengineview, qapp, js_enabled, world, expected):
    if False:
        for i in range(10):
            print('nop')
    'With QtWebEngine, runJavaScript works even when JS is off.'
    from qutebrowser.qt.webenginecore import QWebEngineSettings, QWebEngineScript
    assert world in [QWebEngineScript.ScriptWorldId.MainWorld, QWebEngineScript.ScriptWorldId.ApplicationWorld, QWebEngineScript.ScriptWorldId.UserWorld]
    settings = webengineview.settings()
    settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, js_enabled)
    qapp.processEvents()
    page = webengineview.page()
    with qtbot.wait_callback() as callback:
        page.runJavaScript('1 + 1', world, callback)
    callback.assert_called_with(expected)