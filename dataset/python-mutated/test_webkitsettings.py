import pytest
pytest.importorskip('qutebrowser.qt.webkitwidgets')
from qutebrowser.browser.webkit import webkitsettings

def test_parsed_user_agent(qapp):
    if False:
        for i in range(10):
            print('nop')
    webkitsettings._init_user_agent()
    parsed = webkitsettings.parsed_user_agent
    assert parsed.upstream_browser_key == 'Version'
    assert parsed.qt_key == 'Qt'