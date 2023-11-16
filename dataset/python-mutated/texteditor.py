"""
Text editor dialog
"""
import sys
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout
from spyder.api.config.fonts import SpyderFontsMixin, SpyderFontType
from spyder.config.base import _
from spyder.py3compat import is_binary_string, to_binary_string, to_text_string
from spyder.utils.icon_manager import ima
from spyder.plugins.variableexplorer.widgets.basedialog import BaseDialog

class TextEditor(BaseDialog, SpyderFontsMixin):
    """Array Editor Dialog"""

    def __init__(self, text, title='', parent=None, readonly=False):
        if False:
            return 10
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.text = None
        self.btn_save_and_close = None
        if is_binary_string(text):
            self.is_binary = True
            text = to_text_string(text, 'utf8')
        else:
            self.is_binary = False
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.edit = QTextEdit(parent)
        self.edit.setReadOnly(readonly)
        self.edit.textChanged.connect(self.text_changed)
        self.edit.setPlainText(text)
        font = self.get_font(SpyderFontType.MonospaceInterface)
        self.edit.setFont(font)
        self.layout.addWidget(self.edit)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        if not readonly:
            self.btn_save_and_close = QPushButton(_('Save and Close'))
            self.btn_save_and_close.setDisabled(True)
            self.btn_save_and_close.clicked.connect(self.accept)
            btn_layout.addWidget(self.btn_save_and_close)
        self.btn_close = QPushButton(_('Close'))
        self.btn_close.setAutoDefault(True)
        self.btn_close.setDefault(True)
        self.btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_close)
        self.layout.addLayout(btn_layout)
        if sys.platform == 'darwin':
            self.setWindowFlags(Qt.Tool)
        else:
            self.setWindowFlags(Qt.Window)
        self.setWindowIcon(ima.icon('edit'))
        if title:
            try:
                unicode_title = to_text_string(title)
            except UnicodeEncodeError:
                unicode_title = u''
        else:
            unicode_title = u''
        self.setWindowTitle(_('Text editor') + u'%s' % (u' - ' + unicode_title if unicode_title else u''))

    @Slot()
    def text_changed(self):
        if False:
            print('Hello World!')
        'Text has changed'
        if self.is_binary:
            self.text = to_binary_string(self.edit.toPlainText(), 'utf8')
        else:
            self.text = to_text_string(self.edit.toPlainText())
        if self.btn_save_and_close:
            self.btn_save_and_close.setEnabled(True)
            self.btn_save_and_close.setAutoDefault(True)
            self.btn_save_and_close.setDefault(True)

    def get_value(self):
        if False:
            while True:
                i = 10
        'Return modified text'
        return self.text

    def setup_and_check(self, value):
        if False:
            while True:
                i = 10
        'Verify if TextEditor is able to display strings passed to it.'
        try:
            to_text_string(value, 'utf8')
            return True
        except:
            return False

def test():
    if False:
        while True:
            i = 10
    'Text editor demo'
    from spyder.utils.qthelpers import qapplication
    _app = qapplication()
    text = '01234567890123456789012345678901234567890123456789012345678901234567890123456789\ndedekdh elkd ezd ekjd lekdj elkdfjelfjk e'
    dialog = TextEditor(text)
    dialog.exec_()
    dlg_text = dialog.get_value()
    assert text == dlg_text
if __name__ == '__main__':
    test()