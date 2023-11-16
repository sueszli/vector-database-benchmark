"""This module contains the close quotes editor extension."""
from qtpy.QtGui import QTextCursor
from spyder.plugins.editor.api.editorextension import EditorExtension

class CloseBracketsExtension(EditorExtension):
    """Editor Extension for insert brackets automatically."""
    BRACKETS_CHAR = ['(', ')', '{', '}', '[', ']']
    BRACKETS_LEFT = BRACKETS_CHAR[::2]
    BRACKETS_RIGHT = BRACKETS_CHAR[1::2]
    BRACKETS_PAIR = {'(': '()', ')': '()', '{': '{}', '}': '{}', '[': '[]', ']': '[]'}

    def on_state_changed(self, state):
        if False:
            while True:
                i = 10
        'Connect/disconnect sig_key_pressed signal.'
        if state:
            self.editor.sig_key_pressed.connect(self._on_key_pressed)
        else:
            self.editor.sig_key_pressed.disconnect(self._on_key_pressed)

    def _on_key_pressed(self, event):
        if False:
            i = 10
            return i + 15
        if event.isAccepted():
            return
        char = event.text()
        if char in self.BRACKETS_CHAR and self.enabled:
            self.editor.completion_widget.hide()
            self._autoinsert_brackets(char)
            event.accept()

    def unmatched_brackets_in_line(self, text, closing_brackets_type=None, autoinsert=False):
        if False:
            while True:
                i = 10
        "\n        Checks if there is an unmatched brackets in the 'text'.\n\n        The brackets type can be general or specified by closing_brackets_type\n        (')', ']' or '}')\n        "
        if closing_brackets_type is None:
            opening_brackets = self.BRACKETS_LEFT
            closing_brackets = self.BRACKETS_RIGHT
        else:
            closing_brackets = [closing_brackets_type]
            opening_brackets = [{')': '(', '}': '{', ']': '['}[closing_brackets_type]]
        block = self.editor.textCursor().block()
        line_pos = block.position()
        for (pos, char) in enumerate(text):
            if char in opening_brackets:
                match = self.editor.find_brace_match(line_pos + pos, char, forward=True)
                if match is None or match > line_pos + len(text):
                    return True
            if char in closing_brackets:
                match = self.editor.find_brace_match(line_pos + pos, char, forward=False)
                if match is None or (match < line_pos and (not autoinsert)):
                    return True
        return False

    def _autoinsert_brackets(self, char):
        if False:
            for i in range(10):
                print('nop')
        'Control automatic insertation of brackets in various situations.'
        pair = self.BRACKETS_PAIR[char]
        cursor = self.editor.textCursor()
        trailing_text = self.editor.get_text('cursor', 'eol').strip()
        if self.editor.has_selected_text():
            text = self.editor.get_selected_text()
            self.editor.insert_text('{0}{1}{2}'.format(pair[0], text, pair[1]))
            cursor.movePosition(QTextCursor.Left, QTextCursor.MoveAnchor, 1)
            cursor.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, len(text))
            self.editor.setTextCursor(cursor)
        elif char in self.BRACKETS_LEFT:
            if not trailing_text or trailing_text[0] in self.BRACKETS_RIGHT or trailing_text[0] in [',', ':', ';']:
                self.editor.insert_text(pair)
                cursor.movePosition(QTextCursor.PreviousCharacter)
                self.editor.setTextCursor(cursor)
            else:
                self.editor.insert_text(char)
            if char in self.editor.signature_completion_characters:
                self.editor.request_signature()
        elif char in self.BRACKETS_RIGHT:
            if self.editor.next_char() == char and (not self.editor.textCursor().atBlockEnd()) and (not self.unmatched_brackets_in_line(cursor.block().text(), char, autoinsert=True)):
                cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor, 1)
                cursor.clearSelection()
                self.editor.setTextCursor(cursor)
            else:
                self.editor.insert_text(char)