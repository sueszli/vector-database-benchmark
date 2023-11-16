"""Keychain string displayed in the statusbar."""
from qutebrowser.qt.core import pyqtSlot
from qutebrowser.mainwindow.statusbar import textbase
from qutebrowser.utils import usertypes

class KeyString(textbase.TextBase):
    """Keychain string displayed in the statusbar."""

    @pyqtSlot(usertypes.KeyMode, str)
    def on_keystring_updated(self, _mode, keystr):
        if False:
            print('Hello World!')
        self.setText(keystr)