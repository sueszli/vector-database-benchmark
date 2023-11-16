""" Provides bracket matching for Q[Plain]TextEdit widgets.
"""
from qtpy import QtCore, QtGui, QtWidgets

class BracketMatcher(QtCore.QObject):
    """ Matches square brackets, braces, and parentheses based on cursor
        position.
    """
    _opening_map = {'(': ')', '{': '}', '[': ']'}
    _closing_map = {')': '(', '}': '{', ']': '['}

    def __init__(self, text_edit):
        if False:
            i = 10
            return i + 15
        ' Create a call tip manager that is attached to the specified Qt\n            text edit widget.\n        '
        assert isinstance(text_edit, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit))
        super().__init__()
        self.format = QtGui.QTextCharFormat()
        self.format.setBackground(QtGui.QColor('silver'))
        self._text_edit = text_edit
        text_edit.cursorPositionChanged.connect(self._cursor_position_changed)

    def _find_match(self, position):
        if False:
            print('Hello World!')
        ' Given a valid position in the text document, try to find the\n            position of the matching bracket. Returns -1 if unsuccessful.\n        '
        document = self._text_edit.document()
        start_char = document.characterAt(position)
        search_char = self._opening_map.get(start_char)
        if search_char:
            increment = 1
        else:
            search_char = self._closing_map.get(start_char)
            if search_char:
                increment = -1
            else:
                return -1
        char = start_char
        depth = 0
        while position >= 0 and position < document.characterCount():
            if char == start_char:
                depth += 1
            elif char == search_char:
                depth -= 1
            if depth == 0:
                break
            position += increment
            char = document.characterAt(position)
        else:
            position = -1
        return position

    def _selection_for_character(self, position):
        if False:
            return 10
        ' Convenience method for selecting a character.\n        '
        selection = QtWidgets.QTextEdit.ExtraSelection()
        cursor = self._text_edit.textCursor()
        cursor.setPosition(position)
        cursor.movePosition(QtGui.QTextCursor.NextCharacter, QtGui.QTextCursor.KeepAnchor)
        selection.cursor = cursor
        selection.format = self.format
        return selection

    def _cursor_position_changed(self):
        if False:
            i = 10
            return i + 15
        ' Updates the document formatting based on the new cursor position.\n        '
        self._text_edit.setExtraSelections([])
        cursor = self._text_edit.textCursor()
        if not cursor.hasSelection():
            position = cursor.position() - 1
            match_position = self._find_match(position)
            if match_position != -1:
                extra_selections = [self._selection_for_character(pos) for pos in (position, match_position)]
                self._text_edit.setExtraSelections(extra_selections)