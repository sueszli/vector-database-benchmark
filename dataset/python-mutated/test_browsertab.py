import pytest
from qutebrowser.qt.core import QUrl
from qutebrowser.browser import browsertab

class TestAction:

    def test_run_string_valid(self, qtbot, web_tab):
        if False:
            for i in range(10):
                print('nop')
        url_1 = QUrl('qute://testdata/data/backforward/1.txt')
        url_2 = QUrl('qute://testdata/data/backforward/2.txt')
        with qtbot.wait_signal(web_tab.load_finished):
            web_tab.load_url(url_1)
        with qtbot.wait_signal(web_tab.load_finished):
            web_tab.load_url(url_2)
        assert web_tab.url() == url_2
        with qtbot.wait_signal(web_tab.load_finished):
            web_tab.action.run_string('Back')
        assert web_tab.url() == url_1

    @pytest.mark.parametrize('member', ['blah', 'PermissionUnknown'])
    def test_run_string_invalid(self, qtbot, web_tab, member):
        if False:
            while True:
                i = 10
        with pytest.raises(browsertab.WebTabError, match=f'{member} is not a valid web action!'):
            web_tab.action.run_string(member)