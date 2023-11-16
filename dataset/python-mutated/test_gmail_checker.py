import sys
from importlib import reload
from types import ModuleType
from libqtile.widget import gmail_checker
from test.widgets.conftest import FakeBar

class FakeIMAP(ModuleType):

    class IMAP4_SSL:

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            pass

        def login(self, username, password):
            if False:
                while True:
                    i = 10
            self.username = username
            self.password = password

        def status(self, path, *args, **kwargs):
            if False:
                return 10
            if not (self.username and self.password):
                return (False, None)
            return ('OK', ['("{}" (MESSAGES 10 UNSEEN 2)'.format(path).encode()])

def test_gmail_checker_valid_response(fake_qtile, monkeypatch, fake_window):
    if False:
        return 10
    monkeypatch.setitem(sys.modules, 'imaplib', FakeIMAP('imaplib'))
    reload(gmail_checker)
    gmc = gmail_checker.GmailChecker(username='qtile', password='test')
    fakebar = FakeBar([gmc], window=fake_window)
    gmc._configure(fake_qtile, fakebar)
    text = gmc.poll()
    assert text == 'inbox[10],unseen[2]'

def test_gmail_checker_invalid_response(fake_qtile, monkeypatch, fake_window):
    if False:
        return 10
    monkeypatch.setitem(sys.modules, 'imaplib', FakeIMAP('imaplib'))
    reload(gmail_checker)
    gmc = gmail_checker.GmailChecker()
    fakebar = FakeBar([gmc], window=fake_window)
    gmc._configure(fake_qtile, fakebar)
    text = gmc.poll()
    assert text == 'UNKNOWN ERROR'

def test_gmail_checker_only_unseen(fake_qtile, monkeypatch, fake_window):
    if False:
        while True:
            i = 10
    monkeypatch.setitem(sys.modules, 'imaplib', FakeIMAP('imaplib'))
    reload(gmail_checker)
    gmc = gmail_checker.GmailChecker(display_fmt='unseen[{0}]', status_only_unseen=True, username='qtile', password='test')
    fakebar = FakeBar([gmc], window=fake_window)
    gmc._configure(fake_qtile, fakebar)
    text = gmc.poll()
    assert text == 'unseen[2]'