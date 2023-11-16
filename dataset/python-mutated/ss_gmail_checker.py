import sys
from importlib import reload
import pytest
from libqtile.widget import gmail_checker
from test.widgets.test_gmail_checker import FakeIMAP

@pytest.fixture
def widget(monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setitem(sys.modules, 'imaplib', FakeIMAP('imaplib'))
    reload(gmail_checker)
    yield gmail_checker.GmailChecker

@pytest.mark.parametrize('screenshot_manager', [{'username': 'qtile', 'password': 'qtile'}, {'username': 'qtile', 'password': 'qtile', 'display_fmt': 'unseen[{0}]', 'status_only_unseen': True}], indirect=True)
def ss_gmail_checker(screenshot_manager):
    if False:
        print('Hello World!')
    screenshot_manager.take_screenshot()