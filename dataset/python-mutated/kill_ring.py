""" A generic Emacs-style kill ring, as well as a Qt-specific version.
"""
from qtpy import QtCore, QtWidgets, QtGui

class KillRing(object):
    """ A generic Emacs-style kill ring.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.clear()

    def clear(self):
        if False:
            print('Hello World!')
        ' Clears the kill ring.\n        '
        self._index = -1
        self._ring = []

    def kill(self, text):
        if False:
            i = 10
            return i + 15
        ' Adds some killed text to the ring.\n        '
        self._ring.append(text)

    def yank(self):
        if False:
            for i in range(10):
                print('nop')
        ' Yank back the most recently killed text.\n\n        Returns\n        -------\n        A text string or None.\n        '
        self._index = len(self._ring)
        return self.rotate()

    def rotate(self):
        if False:
            print('Hello World!')
        ' Rotate the kill ring, then yank back the new top.\n\n        Returns\n        -------\n        A text string or None.\n        '
        self._index -= 1
        if self._index >= 0:
            return self._ring[self._index]
        return None

class QtKillRing(QtCore.QObject):
    """ A kill ring attached to Q[Plain]TextEdit.
    """

    def __init__(self, text_edit):
        if False:
            return 10
        ' Create a kill ring attached to the specified Qt text edit.\n        '
        assert isinstance(text_edit, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit))
        super().__init__()
        self._ring = KillRing()
        self._prev_yank = None
        self._skip_cursor = False
        self._text_edit = text_edit
        text_edit.cursorPositionChanged.connect(self._cursor_position_changed)

    def clear(self):
        if False:
            print('Hello World!')
        ' Clears the kill ring.\n        '
        self._ring.clear()
        self._prev_yank = None

    def kill(self, text):
        if False:
            for i in range(10):
                print('nop')
        ' Adds some killed text to the ring.\n        '
        self._ring.kill(text)

    def kill_cursor(self, cursor):
        if False:
            while True:
                i = 10
        ' Kills the text selected by the give cursor.\n        '
        text = cursor.selectedText()
        if text:
            cursor.removeSelectedText()
            self.kill(text)

    def yank(self):
        if False:
            while True:
                i = 10
        ' Yank back the most recently killed text.\n        '
        text = self._ring.yank()
        if text:
            self._skip_cursor = True
            cursor = self._text_edit.textCursor()
            cursor.insertText(text)
            self._prev_yank = text

    def rotate(self):
        if False:
            print('Hello World!')
        ' Rotate the kill ring, then yank back the new top.\n        '
        if self._prev_yank:
            text = self._ring.rotate()
            if text:
                self._skip_cursor = True
                cursor = self._text_edit.textCursor()
                cursor.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, n=len(self._prev_yank))
                cursor.insertText(text)
                self._prev_yank = text

    def _cursor_position_changed(self):
        if False:
            return 10
        if self._skip_cursor:
            self._skip_cursor = False
        else:
            self._prev_yank = None