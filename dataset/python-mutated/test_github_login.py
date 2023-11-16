"""Tests for the Github authentication dialog."""
import pytest
from qtpy.QtCore import Qt
from spyder.widgets.github.gh_login import DlgGitHubLogin

@pytest.fixture
def github_dialog(qtbot):
    if False:
        print('Hello World!')
    'Set up error report dialog.'
    widget = DlgGitHubLogin(None, None)
    qtbot.addWidget(widget)
    return widget

def test_dialog(github_dialog, qtbot):
    if False:
        i = 10
        return i + 15
    'Test that error report dialog UI behaves properly.'
    dlg = github_dialog
    assert not dlg.bt_sign_in.isEnabled()
    qtbot.keyClicks(dlg.le_token, 'token')
    assert dlg.bt_sign_in.isEnabled()
if __name__ == '__main__':
    pytest.main()