import sys
from importlib import reload
from types import ModuleType
import pytest
from test.widgets.conftest import FakeBar

class FakeIMAP(ModuleType):

    class IMAP4_SSL:

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
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
            return ('OK', ['"{}" (UNSEEN 2)'.format(path).encode()])

        def logout(self):
            if False:
                i = 10
                return i + 15
            pass

class FakeKeyring(ModuleType):
    valid = True
    error = True

    def get_password(self, _app, user):
        if False:
            while True:
                i = 10
        if self.valid:
            return 'password'
        else:
            if self.error:
                return 'Gnome Keyring Error'
            return None

@pytest.fixture()
def patched_imap(monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.delitem(sys.modules, 'imaplib', raising=False)
    monkeypatch.delitem(sys.modules, 'keyring', raising=False)
    monkeypatch.setitem(sys.modules, 'imaplib', FakeIMAP('imaplib'))
    monkeypatch.setitem(sys.modules, 'keyring', FakeKeyring('keyring'))
    from libqtile.widget import imapwidget
    reload(imapwidget)
    yield imapwidget

def test_imapwidget(fake_qtile, monkeypatch, fake_window, patched_imap):
    if False:
        print('Hello World!')
    imap = patched_imap.ImapWidget(user='qtile')
    fakebar = FakeBar([imap], window=fake_window)
    imap._configure(fake_qtile, fakebar)
    text = imap.poll()
    assert text == 'INBOX: 2'

def test_imapwidget_keyring_error(fake_qtile, monkeypatch, fake_window, patched_imap):
    if False:
        i = 10
        return i + 15
    patched_imap.keyring.valid = False
    imap = patched_imap.ImapWidget(user='qtile')
    fakebar = FakeBar([imap], window=fake_window)
    imap._configure(fake_qtile, fakebar)
    text = imap.poll()
    assert text == 'Gnome Keyring Error'

def test_imapwidget_password_none(fake_qtile, monkeypatch, fake_window, patched_imap):
    if False:
        i = 10
        return i + 15
    patched_imap.keyring.valid = False
    patched_imap.keyring.error = False
    imap = patched_imap.ImapWidget(user='qtile')
    fakebar = FakeBar([imap], window=fake_window)
    imap._configure(fake_qtile, fakebar)
    with pytest.raises(AttributeError):
        with pytest.raises(UnboundLocalError):
            imap.poll()