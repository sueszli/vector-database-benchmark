import re
import unicodedata
from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QCursor, QKeySequence, QTextCursor
from PyQt6.QtWidgets import QCompleter, QTextEdit, QToolTip
from picard.config import BoolOption, get_config
from picard.const.sys import IS_MACOS
from picard.script import ScriptFunctionDocError, script_function_documentation, script_function_names
from picard.util.tags import PRESERVED_TAGS, TAG_NAMES, display_tag_name
from picard.ui import FONT_FAMILY_MONOSPACE
from picard.ui.theme import theme
EXTRA_VARIABLES = ('~absolutetracknumber', '~albumartists_sort', '~albumartists', '~artists_sort', '~datatrack', '~discpregap', '~multiartist', '~musicbrainz_discids', '~musicbrainz_tracknumber', '~performance_attributes', '~pregap', '~primaryreleasetype', '~rating', '~recording_firstreleasedate', '~recordingcomment', '~recordingtitle', '~releasecomment', '~releasecountries', '~releasegroup_firstreleasedate', '~releasegroup', '~releasegroupcomment', '~releaselanguage', '~secondaryreleasetype', '~silence', '~totalalbumtracks', '~video')

def find_regex_index(regex, text, start=0):
    if False:
        return 10
    match = regex.search(text[start:])
    return start + match.start() if match else -1

class TaggerScriptSyntaxHighlighter(QtGui.QSyntaxHighlighter):

    def __init__(self, document):
        if False:
            while True:
                i = 10
        super().__init__(document)
        syntax_theme = theme.syntax_theme
        self.func_re = re.compile('\\$(?!noop)[_a-zA-Z0-9]*\\(')
        self.func_fmt = QtGui.QTextCharFormat()
        self.func_fmt.setFontWeight(QtGui.QFont.Weight.Bold)
        self.func_fmt.setForeground(syntax_theme.func)
        self.var_re = re.compile('%[_a-zA-Z0-9:]*%')
        self.var_fmt = QtGui.QTextCharFormat()
        self.var_fmt.setForeground(syntax_theme.var)
        self.unicode_re = re.compile('\\\\u[a-fA-F0-9]{4}')
        self.unicode_fmt = QtGui.QTextCharFormat()
        self.unicode_fmt.setForeground(syntax_theme.escape)
        self.escape_re = re.compile('\\\\[^u]')
        self.escape_fmt = QtGui.QTextCharFormat()
        self.escape_fmt.setForeground(syntax_theme.escape)
        self.special_re = re.compile('[^\\\\][(),]')
        self.special_fmt = QtGui.QTextCharFormat()
        self.special_fmt.setForeground(syntax_theme.special)
        self.bracket_re = re.compile('[()]')
        self.noop_re = re.compile('\\$noop\\(')
        self.noop_fmt = QtGui.QTextCharFormat()
        self.noop_fmt.setFontWeight(QtGui.QFont.Weight.Bold)
        self.noop_fmt.setFontItalic(True)
        self.noop_fmt.setForeground(syntax_theme.noop)
        self.rules = [(self.func_re, self.func_fmt, 0, -1), (self.var_re, self.var_fmt, 0, 0), (self.unicode_re, self.unicode_fmt, 0, 0), (self.escape_re, self.escape_fmt, 0, 0), (self.special_re, self.special_fmt, 1, -1)]

    def highlightBlock(self, text):
        if False:
            while True:
                i = 10
        self.setCurrentBlockState(0)
        for (expr, fmt, a, b) in self.rules:
            for match in expr.finditer(text):
                index = match.start()
                length = match.end() - match.start()
                self.setFormat(index + a, length + b, fmt)
        index = find_regex_index(self.noop_re, text) if self.previousBlockState() <= 0 else 0
        open_brackets = self.previousBlockState() if self.previousBlockState() > 0 else 0
        text_length = len(text)
        while index >= 0:
            next_index = find_regex_index(self.bracket_re, text, index)
            if next_index > 0 and text[next_index - 1] == '\\':
                next_index += 1
            if next_index >= text_length:
                self.setFormat(index, text_length - index, self.noop_fmt)
                break
            if next_index > -1 and text[next_index] == '(':
                open_brackets += 1
            elif next_index > -1 and text[next_index] == ')':
                open_brackets -= 1
            if next_index > -1:
                self.setFormat(index, next_index - index + 1, self.noop_fmt)
            elif next_index == -1 and open_brackets > 0:
                self.setFormat(index, text_length - index, self.noop_fmt)
            if open_brackets == 0:
                next_index = find_regex_index(self.noop_re, text, next_index)
            index = next_index + 1 if next_index > -1 and next_index < text_length else -1
        self.setCurrentBlockState(open_brackets)

class ScriptCompleter(QCompleter):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(sorted(self.choices), parent)
        self.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self.highlighted.connect(self.set_highlighted)
        self.last_selected = ''

    @property
    def choices(self):
        if False:
            return 10
        yield from {'$' + name for name in script_function_names()}
        yield from {'%' + name.replace('~', '_') + '%' for name in self.all_tags}

    @property
    def all_tags(self):
        if False:
            for i in range(10):
                print('nop')
        yield from TAG_NAMES.keys()
        yield from PRESERVED_TAGS
        yield from EXTRA_VARIABLES

    def set_highlighted(self, text):
        if False:
            while True:
                i = 10
        self.last_selected = text

    def get_selected(self):
        if False:
            while True:
                i = 10
        return self.last_selected

class DocumentedScriptToken:
    allowed_chars = re.compile('[A-Za-z0-9_]')

    def __init__(self, doc, cursor_position):
        if False:
            print('Hello World!')
        self._doc = doc
        self._cursor_position = cursor_position

    def is_start_char(self, char):
        if False:
            i = 10
            return i + 15
        return False

    def is_allowed_char(self, char, position):
        if False:
            print('Hello World!')
        return self.allowed_chars.match(char)

    def get_tooltip(self, position):
        if False:
            return 10
        return None

    def _read_text(self, position, count):
        if False:
            while True:
                i = 10
        text = ''
        while count:
            char = self._doc.characterAt(position)
            if not char:
                break
            text += char
            count -= 1
            position += 1
        return text

    def _read_allowed_chars(self, position):
        if False:
            i = 10
            return i + 15
        doc = self._doc
        text = ''
        while True:
            char = doc.characterAt(position)
            if not self.allowed_chars.match(char):
                break
            text += char
            position += 1
        return text

class FunctionScriptToken(DocumentedScriptToken):

    def is_start_char(self, char):
        if False:
            for i in range(10):
                print('nop')
        return char == '$'

    def get_tooltip(self, position):
        if False:
            for i in range(10):
                print('nop')
        if self._doc.characterAt(position) != '$':
            return None
        function = self._read_allowed_chars(position + 1)
        try:
            return script_function_documentation(function, 'html')
        except ScriptFunctionDocError:
            return None

class VariableScriptToken(DocumentedScriptToken):
    allowed_chars = re.compile('[A-Za-z0-9_:]')

    def is_start_char(self, char):
        if False:
            i = 10
            return i + 15
        return char == '%'

    def get_tooltip(self, position):
        if False:
            i = 10
            return i + 15
        if self._doc.characterAt(position) != '%':
            return None
        tag = self._read_allowed_chars(position + 1)
        return display_tag_name(tag)

class UnicodeEscapeScriptToken(DocumentedScriptToken):
    allowed_chars = re.compile('[uA-Fa-f0-9]')
    unicode_escape_sequence = re.compile('^\\\\u[a-fA-F0-9]{4}$')

    def is_start_char(self, char):
        if False:
            while True:
                i = 10
        return char == '\\'

    def is_allowed_char(self, char, position):
        if False:
            i = 10
            return i + 15
        return self.allowed_chars.match(char) and self._cursor_position - position < 6

    def get_tooltip(self, position):
        if False:
            i = 10
            return i + 15
        text = self._read_text(position, 6)
        if self.unicode_escape_sequence.match(text):
            codepoint = int(text[2:], 16)
            char = chr(codepoint)
            try:
                tooltip = unicodedata.name(char)
            except ValueError:
                tooltip = f'U+{text[2:].upper()}'
            if unicodedata.category(char)[0] != 'C':
                tooltip += f': "{char}"'
            return tooltip
        return None

def _clean_text(text):
    if False:
        i = 10
        return i + 15
    return ''.join(_replace_control_chars(text))

def _replace_control_chars(text):
    if False:
        while True:
            i = 10
    simple_ctrl_chars = {'\n', '\r', '\t'}
    for ch in text:
        if ch not in simple_ctrl_chars and unicodedata.category(ch)[0] == 'C':
            yield ('\\u' + hex(ord(ch))[2:].rjust(4, '0'))
        else:
            yield ch

class ScriptTextEdit(QTextEdit):
    autocomplete_trigger_chars = re.compile('[$%A-Za-z0-9_]')
    options = [BoolOption('persist', 'script_editor_wordwrap', False), BoolOption('persist', 'script_editor_tooltips', True)]

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        config = get_config()
        self.highlighter = TaggerScriptSyntaxHighlighter(self.document())
        self.enable_completer()
        self.setFontFamily(FONT_FAMILY_MONOSPACE)
        self.setMouseTracking(True)
        self.setAcceptRichText(False)
        self.wordwrap_action = QAction(_('&Word wrap script'), self)
        self.wordwrap_action.setToolTip(_('Word wrap long lines in the editor'))
        self.wordwrap_action.triggered.connect(self.update_wordwrap)
        self.wordwrap_action.setShortcut(QKeySequence(_('Ctrl+Shift+W')))
        self.wordwrap_action.setCheckable(True)
        self.wordwrap_action.setChecked(config.persist['script_editor_wordwrap'])
        self.update_wordwrap()
        self.addAction(self.wordwrap_action)
        self._show_tooltips = config.persist['script_editor_tooltips']
        self.show_tooltips_action = QAction(_('Show help &tooltips'), self)
        self.show_tooltips_action.setToolTip(_('Show tooltips for script elements'))
        self.show_tooltips_action.triggered.connect(self.update_show_tooltips)
        self.show_tooltips_action.setShortcut(QKeySequence(_('Ctrl+Shift+T')))
        self.show_tooltips_action.setCheckable(True)
        self.show_tooltips_action.setChecked(self._show_tooltips)
        self.addAction(self.show_tooltips_action)
        self.textChanged.connect(self.update_tooltip)

    def contextMenuEvent(self, event):
        if False:
            print('Hello World!')
        menu = self.createStandardContextMenu()
        menu.addSeparator()
        menu.addAction(self.wordwrap_action)
        menu.addAction(self.show_tooltips_action)
        menu.exec(event.globalPos())

    def mouseMoveEvent(self, event):
        if False:
            i = 10
            return i + 15
        if self._show_tooltips:
            tooltip = self.get_tooltip_at_mouse_position(event.pos())
            if not tooltip:
                QToolTip.hideText()
            self.setToolTip(tooltip)
        return super().mouseMoveEvent(event)

    def update_tooltip(self):
        if False:
            for i in range(10):
                print('nop')
        if self.underMouse() and self.toolTip():
            position = self.mapFromGlobal(QCursor.pos())
            tooltip = self.get_tooltip_at_mouse_position(position)
            if tooltip != self.toolTip():
                QToolTip.hideText()
                self.setToolTip(tooltip)

    def get_tooltip_at_mouse_position(self, position):
        if False:
            print('Hello World!')
        cursor = self.cursorForPosition(position)
        return self.get_tooltip_at_cursor(cursor)

    def get_tooltip_at_cursor(self, cursor):
        if False:
            print('Hello World!')
        position = cursor.position()
        doc = self.document()
        documented_tokens = {FunctionScriptToken(doc, position), VariableScriptToken(doc, position), UnicodeEscapeScriptToken(doc, position)}
        while position >= 0 and documented_tokens:
            char = doc.characterAt(position)
            for token in list(documented_tokens):
                if token.is_start_char(char):
                    return token.get_tooltip(position)
                elif not token.is_allowed_char(char, position):
                    documented_tokens.remove(token)
            position -= 1
        return None

    def insertFromMimeData(self, source):
        if False:
            while True:
                i = 10
        text = _clean_text(source.text())
        source = QtCore.QMimeData()
        source.setText(text)
        return super().insertFromMimeData(source)

    def setPlainText(self, text):
        if False:
            for i in range(10):
                print('nop')
        super().setPlainText(text)
        self.update_wordwrap()

    def update_wordwrap(self):
        if False:
            while True:
                i = 10
        'Toggles wordwrap in the script editor\n        '
        wordwrap = self.wordwrap_action.isChecked()
        config = get_config()
        config.persist['script_editor_wordwrap'] = wordwrap
        if wordwrap:
            self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        else:
            self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

    def update_show_tooltips(self):
        if False:
            for i in range(10):
                print('nop')
        'Toggles wordwrap in the script editor\n        '
        self._show_tooltips = self.show_tooltips_action.isChecked()
        config = get_config()
        config.persist['script_editor_tooltips'] = self._show_tooltips
        if not self._show_tooltips:
            QToolTip.hideText()
            self.setToolTip('')

    def enable_completer(self):
        if False:
            i = 10
            return i + 15
        self.completer = ScriptCompleter()
        self.completer.setWidget(self)
        self.completer.activated.connect(self.insert_completion)
        self.popup_shown = False

    def insert_completion(self, completion):
        if False:
            for i in range(10):
                print('nop')
        if not completion:
            return
        tc = self.cursor_select_word()
        if completion.startswith('$'):
            completion += '('
        tc.insertText(completion)
        if not tc.atEnd():
            pos = tc.position()
            tc = self.textCursor()
            tc.setPosition(pos + 1, QTextCursor.MoveMode.KeepAnchor)
            first_char = completion[0]
            next_char = tc.selectedText()
            if first_char == '$' and next_char == '(' or (first_char == '%' and next_char == '%'):
                tc.removeSelectedText()
            else:
                tc.setPosition(pos)
        self.setTextCursor(tc)
        self.popup_hide()

    def popup_hide(self):
        if False:
            while True:
                i = 10
        self.completer.popup().hide()

    def cursor_select_word(self, full_word=True):
        if False:
            return 10
        tc = self.textCursor()
        current_position = tc.position()
        tc.select(QTextCursor.SelectionType.WordUnderCursor)
        selected_text = tc.selectedText()
        if current_position > 0 and selected_text and (selected_text[0] in {'(', '%'}):
            current_position -= 1
            tc.setPosition(current_position)
            tc.select(QTextCursor.SelectionType.WordUnderCursor)
            selected_text = tc.selectedText()
        start = tc.selectionStart()
        end = tc.selectionEnd()
        if current_position < start or current_position > end:
            tc.setPosition(current_position)
            selected_text = tc.selectedText()
        if not selected_text.startswith('$') and (not selected_text.startswith('%')):
            tc.setPosition(start - 1 if start > 0 else 0)
            tc.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            selected_text = tc.selectedText()
            if not selected_text.startswith('$') and (not selected_text.startswith('%')):
                tc.setPosition(start)
                tc.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        if not full_word:
            tc.setPosition(current_position, QTextCursor.MoveMode.KeepAnchor)
        return tc

    def keyPressEvent(self, event):
        if False:
            while True:
                i = 10
        if self.completer.popup().isVisible():
            if event.key() in {Qt.Key.Key_Tab, Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                self.completer.activated.emit(self.completer.get_selected())
                return
        super().keyPressEvent(event)
        self.handle_autocomplete(event)

    def handle_autocomplete(self, event):
        if False:
            print('Hello World!')
        modifier = QtCore.Qt.KeyboardModifier.MetaModifier if IS_MACOS else QtCore.Qt.KeyboardModifier.ControlModifier
        force_completion_popup = event.key() == QtCore.Qt.Key.Key_Space and event.modifiers() & modifier
        if not (force_completion_popup or event.key() in {Qt.Key.Key_Backspace, Qt.Key.Key_Delete} or self.autocomplete_trigger_chars.match(event.text())):
            self.popup_hide()
            return
        tc = self.cursor_select_word(full_word=False)
        selected_text = tc.selectedText()
        if force_completion_popup or (selected_text and selected_text[0] in {'$', '%'}):
            self.completer.setCompletionPrefix(selected_text)
            popup = self.completer.popup()
            popup.setCurrentIndex(self.completer.currentIndex())
            cr = self.cursorRect()
            cr.setWidth(popup.sizeHintForColumn(0) + popup.verticalScrollBar().sizeHint().width())
            self.completer.complete(cr)
        else:
            self.popup_hide()