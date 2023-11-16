"""This module contains the close quotes editor extension."""
import re
from qtpy.QtGui import QTextCursor
from spyder.plugins.editor.api.editorextension import EditorExtension

def unmatched_quotes_in_line(text):
    if False:
        for i in range(10):
            print('nop')
    'Return whether a string has open quotes.\n\n    This simply counts whether the number of quote characters of either\n    type in the string is odd.\n\n    Take from the IPython project (in IPython/core/completer.py in v0.13)\n    Spyder team: Add some changes to deal with escaped quotes\n\n    - Copyright (C) 2008-2011 IPython Development Team\n    - Copyright (C) 2001-2007 Fernando Perez. <fperez@colorado.edu>\n    - Copyright (C) 2001 Python Software Foundation, www.python.org\n\n    Distributed under the terms of the BSD License.\n    '
    text = re.sub("(?<!\\\\)\\\\'", '', text)
    text = re.sub('(?<!\\\\)\\\\"', '', text)
    if text.count('"') % 2:
        return '"'
    elif text.count("'") % 2:
        return "'"
    else:
        return ''

class CloseQuotesExtension(EditorExtension):
    """Editor Extension for insert closing quotes automatically."""

    def on_state_changed(self, state):
        if False:
            print('Hello World!')
        'Connect/disconnect sig_key_pressed signal.'
        if state:
            self.editor.sig_key_pressed.connect(self._on_key_pressed)
        else:
            self.editor.sig_key_pressed.disconnect(self._on_key_pressed)

    def _on_key_pressed(self, event):
        if False:
            while True:
                i = 10
        if event.isAccepted():
            return
        char = event.text()
        if char in ('"', "'") and self.enabled:
            self.editor.completion_widget.hide()
            self._autoinsert_quotes(char)
            event.accept()

    def _autoinsert_quotes(self, char):
        if False:
            while True:
                i = 10
        'Control how to automatically insert quotes in various situations.'
        line_text = self.editor.get_text('sol', 'eol')
        line_to_cursor = self.editor.get_text('sol', 'cursor')
        cursor = self.editor.textCursor()
        last_three = self.editor.get_text('sol', 'cursor')[-3:]
        last_two = self.editor.get_text('sol', 'cursor')[-2:]
        trailing_text = self.editor.get_text('cursor', 'eol').strip()
        if self.editor.has_selected_text():
            text = self.editor.get_selected_text()
            self.editor.insert_text('{0}{1}{0}'.format(char, text))
            cursor.movePosition(QTextCursor.Left, QTextCursor.MoveAnchor, 1)
            cursor.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, len(text))
            self.editor.setTextCursor(cursor)
        elif self.editor.in_comment():
            self.editor.insert_text(char)
        elif len(trailing_text) > 0 and (not unmatched_quotes_in_line(line_to_cursor) == char) and (not trailing_text[0] in (',', ':', ';', ')', ']', '}')):
            self.editor.insert_text(char)
        elif unmatched_quotes_in_line(line_text) and (not last_three == 3 * char):
            self.editor.insert_text(char)
        elif self.editor.next_char() == char:
            cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor, 1)
            cursor.clearSelection()
            self.editor.setTextCursor(cursor)
        elif last_three == 3 * char:
            self.editor.insert_text(3 * char)
            cursor = self.editor.textCursor()
            cursor.movePosition(QTextCursor.PreviousCharacter, QTextCursor.KeepAnchor, 3)
            cursor.clearSelection()
            self.editor.setTextCursor(cursor)
        elif last_two == 2 * char:
            self.editor.insert_text(char)
            self.editor.delayed_popup_docstring()
        else:
            self.editor.insert_text(2 * char)
            cursor = self.editor.textCursor()
            cursor.movePosition(QTextCursor.PreviousCharacter)
            self.editor.setTextCursor(cursor)