from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCompleter, QPlainTextEdit, QApplication
from .util import ButtonsTextEdit

class CompletionTextEdit(ButtonsTextEdit):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        ButtonsTextEdit.__init__(self)
        self.completer = None
        self.moveCursor(QTextCursor.End)
        self.disable_suggestions()

    def set_completer(self, completer):
        if False:
            i = 10
            return i + 15
        self.completer = completer
        self.initialize_completer()

    def initialize_completer(self):
        if False:
            for i in range(10):
                print('nop')
        self.completer.setWidget(self)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.completer.activated.connect(self.insert_completion)
        self.enable_suggestions()

    def insert_completion(self, completion):
        if False:
            while True:
                i = 10
        if self.completer.widget() != self:
            return
        text_cursor = self.textCursor()
        extra = len(completion) - len(self.completer.completionPrefix())
        text_cursor.movePosition(QTextCursor.Left)
        text_cursor.movePosition(QTextCursor.EndOfWord)
        if extra == 0:
            text_cursor.insertText(' ')
        else:
            text_cursor.insertText(completion[-extra:] + ' ')
        self.setTextCursor(text_cursor)

    def text_under_cursor(self):
        if False:
            i = 10
            return i + 15
        tc = self.textCursor()
        tc.select(QTextCursor.WordUnderCursor)
        return tc.selectedText()

    def enable_suggestions(self):
        if False:
            return 10
        self.suggestions_enabled = True

    def disable_suggestions(self):
        if False:
            while True:
                i = 10
        self.suggestions_enabled = False

    def keyPressEvent(self, e):
        if False:
            print('Hello World!')
        if self.isReadOnly():
            return
        if self.is_special_key(e):
            e.ignore()
            return
        QPlainTextEdit.keyPressEvent(self, e)
        if self.isReadOnly():
            return
        ctrlOrShift = bool(int(e.modifiers()) & int(Qt.ControlModifier | Qt.ShiftModifier))
        if self.completer is None or (ctrlOrShift and (not e.text())):
            return
        if not self.suggestions_enabled:
            return
        eow = '~!@#$%^&*()_+{}|:"<>?,./;\'[]\\-='
        hasModifier = e.modifiers() != Qt.NoModifier and (not ctrlOrShift)
        completionPrefix = self.text_under_cursor()
        if hasModifier or not e.text() or len(completionPrefix) < 1 or (eow.find(e.text()[-1]) >= 0):
            self.completer.popup().hide()
            return
        if completionPrefix != self.completer.completionPrefix():
            self.completer.setCompletionPrefix(completionPrefix)
            self.completer.popup().setCurrentIndex(self.completer.completionModel().index(0, 0))
        cr = self.cursorRect()
        cr.setWidth(self.completer.popup().sizeHintForColumn(0) + self.completer.popup().verticalScrollBar().sizeHint().width())
        self.completer.complete(cr)

    def is_special_key(self, e):
        if False:
            while True:
                i = 10
        if self.completer and self.completer.popup().isVisible():
            if e.key() in (Qt.Key_Enter, Qt.Key_Return):
                return True
        if e.key() == Qt.Key_Tab:
            return True
        return False
if __name__ == '__main__':
    app = QApplication([])
    completer = QCompleter(['alabama', 'arkansas', 'avocado', 'breakfast', 'sausage'])
    te = CompletionTextEdit()
    te.set_completer(completer)
    te.show()
    app.exec_()