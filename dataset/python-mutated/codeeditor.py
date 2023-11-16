"""
Editor widget based on QtGui.QPlainTextEdit
"""
from unicodedata import category
import logging
import os
import os.path as osp
import re
import sre_constants
import sys
import textwrap
from IPython.core.inputtransformer2 import TransformerManager
from packaging.version import parse
from qtpy import QT_VERSION
from qtpy.compat import to_qvariant
from qtpy.QtCore import QEvent, QRegExp, Qt, QTimer, QUrl, Signal, Slot
from qtpy.QtGui import QColor, QCursor, QFont, QKeySequence, QPaintEvent, QPainter, QMouseEvent, QTextCursor, QDesktopServices, QKeyEvent, QTextDocument, QTextFormat, QTextOption, QTextCharFormat, QTextLayout
from qtpy.QtWidgets import QApplication, QMenu, QMessageBox, QSplitter, QScrollBar
from spyder_kernels.utils.dochelpers import getobj
from spyder.config.base import _, running_under_pytest
from spyder.plugins.editor.api.decoration import TextDecoration
from spyder.plugins.editor.api.panel import Panel
from spyder.plugins.editor.extensions import CloseBracketsExtension, CloseQuotesExtension, DocstringWriterExtension, QMenuOnlyForEnter, EditorExtensionsManager, SnippetsExtension
from spyder.plugins.completion.api import DiagnosticSeverity
from spyder.plugins.editor.panels import ClassFunctionDropdown, EdgeLine, FoldingPanel, IndentationGuide, LineNumberArea, PanelsManager, ScrollFlagArea
from spyder.plugins.editor.utils.editor import TextHelper, BlockUserData, get_file_language
from spyder.plugins.editor.utils.kill_ring import QtKillRing
from spyder.plugins.editor.utils.languages import ALL_LANGUAGES, CELL_LANGUAGES
from spyder.plugins.editor.widgets.gotoline import GoToLineDialog
from spyder.plugins.editor.widgets.base import TextEditBaseWidget
from spyder.plugins.editor.widgets.codeeditor.lsp_mixin import LSPMixin
from spyder.plugins.outlineexplorer.api import OutlineExplorerData as OED, is_cell_header
from spyder.py3compat import to_text_string, is_string
from spyder.utils import encoding, sourcecode
from spyder.utils.clipboard_helper import CLIPBOARD_HELPER
from spyder.utils.icon_manager import ima
from spyder.utils import syntaxhighlighters as sh
from spyder.utils.palette import SpyderPalette, QStylePalette
from spyder.utils.qthelpers import add_actions, create_action, file_uri, mimedata2url, start_file
from spyder.utils.vcs import get_git_remotes, remote_to_url
from spyder.utils.qstringhelpers import qstring_length
try:
    import nbformat as nbformat
    from nbconvert import PythonExporter as nbexporter
except Exception:
    nbformat = None
logger = logging.getLogger(__name__)

class CodeEditor(LSPMixin, TextEditBaseWidget):
    """Source Code Editor Widget based exclusively on Qt"""
    CONF_SECTION = 'editor'
    LANGUAGES = {'Python': (sh.PythonSH, '#'), 'IPython': (sh.IPythonSH, '#'), 'Cython': (sh.CythonSH, '#'), 'Fortran77': (sh.Fortran77SH, 'c'), 'Fortran': (sh.FortranSH, '!'), 'Idl': (sh.IdlSH, ';'), 'Diff': (sh.DiffSH, ''), 'GetText': (sh.GetTextSH, '#'), 'Nsis': (sh.NsisSH, '#'), 'Html': (sh.HtmlSH, ''), 'Yaml': (sh.YamlSH, '#'), 'Cpp': (sh.CppSH, '//'), 'OpenCL': (sh.OpenCLSH, '//'), 'Enaml': (sh.EnamlSH, '#'), 'Markdown': (sh.MarkdownSH, '#'), 'None': (sh.TextSH, '')}
    TAB_ALWAYS_INDENTS = ('py', 'pyw', 'python', 'ipy', 'c', 'cpp', 'cl', 'h', 'pyt', 'pyi')
    UPDATE_DECORATIONS_TIMEOUT = 500
    painted = Signal(QPaintEvent)
    edge_line = None
    indent_guides = None
    sig_filename_changed = Signal(str)
    sig_bookmarks_changed = Signal()
    go_to_definition = Signal(str, int, int)
    sig_show_object_info = Signal(bool)
    sig_cursor_position_changed = Signal(int, int)
    sig_new_file = Signal(str)
    sig_refresh_formatting = Signal(bool)
    sig_focus_changed = Signal()
    sig_key_pressed = Signal(QKeyEvent)
    sig_key_released = Signal(QKeyEvent)
    sig_alt_left_mouse_pressed = Signal(QMouseEvent)
    sig_alt_mouse_moved = Signal(QMouseEvent)
    sig_leave_out = Signal()
    sig_flags_changed = Signal()
    sig_theme_colors_changed = Signal(dict)
    new_text_set = Signal()
    sig_uri_found = Signal(str)
    sig_file_uri_preprocessed = Signal(str)
    '\n    This signal is emitted when the go to uri for a file has been\n    preprocessed.\n\n    Parameters\n    ----------\n    fpath: str\n        The preprocessed file path.\n    '
    sig_show_completion_object_info = Signal(str, str, bool)
    sig_text_was_inserted = Signal()
    sig_will_insert_text = Signal(str)
    sig_will_remove_selection = Signal(tuple, tuple)
    sig_will_paste_text = Signal(str)
    sig_undo = Signal()
    sig_redo = Signal()
    sig_font_changed = Signal()
    sig_save_requested = Signal()

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent=parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.current_project_path = None
        self.setCursorWidth(self.get_conf('cursor/width', section='main'))
        self.text_helper = TextHelper(self)
        self._panels = PanelsManager(self)
        self.tooltip_widget.sig_help_requested.connect(self.show_object_info)
        self.tooltip_widget.sig_completion_help_requested.connect(self.show_completion_object_info)
        self._last_point = None
        self._last_hover_word = None
        self._last_hover_cursor = None
        self._timer_mouse_moving = QTimer(self)
        self._timer_mouse_moving.setInterval(350)
        self._timer_mouse_moving.setSingleShot(True)
        self._timer_mouse_moving.timeout.connect(self._handle_hover)
        self._last_key_pressed_text = ''
        self._last_pressed_key = None
        self._completions_hint_idle = False
        self._timer_completions_hint = QTimer(self)
        self._timer_completions_hint.setSingleShot(True)
        self._timer_completions_hint.timeout.connect(self._set_completions_hint_idle)
        self.completion_widget.sig_completion_hint.connect(self.show_hint_for_completion)
        self._last_hover_pattern_key = None
        self._last_hover_pattern_text = None
        self.edge_line = self.panels.register(EdgeLine(), Panel.Position.FLOATING)
        self.indent_guides = self.panels.register(IndentationGuide(), Panel.Position.FLOATING)
        self.blanks_enabled = False
        self.underline_errors_enabled = False
        self.scrollpastend_enabled = False
        self.background = QColor('white')
        self.panels.register(FoldingPanel())
        self.linenumberarea = self.panels.register(LineNumberArea())
        self.classfuncdropdown = self.panels.register(ClassFunctionDropdown(), Panel.Position.TOP)
        self.occurrence_color = None
        self.ctrl_click_color = None
        self.sideareas_color = None
        self.matched_p_color = None
        self.unmatched_p_color = None
        self.normal_color = None
        self.comment_color = None
        self.highlighter_class = sh.TextSH
        self.highlighter = None
        ccs = 'Spyder'
        if ccs not in sh.COLOR_SCHEME_NAMES:
            ccs = sh.COLOR_SCHEME_NAMES[0]
        self.color_scheme = ccs
        self.highlight_current_line_enabled = False
        self.setVerticalScrollBar(QScrollBar())
        self.warning_color = SpyderPalette.COLOR_WARN_2
        self.error_color = SpyderPalette.COLOR_ERROR_1
        self.todo_color = SpyderPalette.GROUP_9
        self.breakpoint_color = SpyderPalette.ICON_3
        self.occurrence_color = QColor(SpyderPalette.GROUP_2).lighter(160)
        self.found_results_color = QColor(SpyderPalette.COLOR_OCCURRENCE_4)
        self.scrollflagarea = self.panels.register(ScrollFlagArea(), Panel.Position.RIGHT)
        self.panels.refresh()
        self.document_id = id(self)
        self.cursorPositionChanged.connect(self.__cursor_position_changed)
        self.__find_first_pos = None
        self.__find_args = {}
        self.language = None
        self.supported_language = False
        self.supported_cell_language = False
        self.comment_string = None
        self._kill_ring = QtKillRing(self)
        self.blockCountChanged.connect(self.update_bookmarks)
        self.timer_syntax_highlight = QTimer(self)
        self.timer_syntax_highlight.setSingleShot(True)
        self.timer_syntax_highlight.timeout.connect(self.run_pygments_highlighter)
        self.occurrence_highlighting = None
        self.occurrence_timer = QTimer(self)
        self.occurrence_timer.setSingleShot(True)
        self.occurrence_timer.setInterval(1500)
        self.occurrence_timer.timeout.connect(self.mark_occurrences)
        self.occurrences = []
        self.update_decorations_timer = QTimer(self)
        self.update_decorations_timer.setSingleShot(True)
        self.update_decorations_timer.setInterval(self.UPDATE_DECORATIONS_TIMEOUT)
        self.update_decorations_timer.timeout.connect(self.update_decorations)
        self.verticalScrollBar().valueChanged.connect(lambda value: self.update_decorations_timer.start())
        self.textChanged.connect(self._schedule_document_did_change)
        self.textChanged.connect(self.__text_has_changed)
        self.found_results = []
        self.writer_docstring = DocstringWriterExtension(self)
        self.gotodef_action = None
        self.setup_context_menu()
        self.tab_indents = None
        self.tab_mode = True
        self.intelligent_backspace = True
        self.automatic_completions = True
        self.automatic_completions_after_chars = 3
        self.completions_hint = True
        self.completions_hint_after_ms = 500
        self.close_parentheses_enabled = True
        self.close_quotes_enabled = False
        self.add_colons_enabled = True
        self.auto_unindent_enabled = True
        self.setMouseTracking(True)
        self.__cursor_changed = False
        self._mouse_left_button_pressed = False
        self.ctrl_click_color = QColor(Qt.blue)
        self._bookmarks_blocks = {}
        self.bookmarks = []
        self.shortcuts = self.create_shortcuts()
        self.__visible_blocks = []
        self.painted.connect(self._draw_editor_cell_divider)
        self.last_change_position = None
        self.last_position = None
        self.last_auto_indent = None
        self.skip_rstrip = False
        self.strip_trailing_spaces_on_modify = True
        self.hover_hints_enabled = None
        self.editor_extensions = EditorExtensionsManager(self)
        self.editor_extensions.add(CloseQuotesExtension())
        self.editor_extensions.add(SnippetsExtension())
        self.editor_extensions.add(CloseBracketsExtension())
        self.is_undoing = False
        self.is_redoing = False
        self._rehighlight_timer = QTimer(self)
        self._rehighlight_timer.setSingleShot(True)
        self._rehighlight_timer.setInterval(150)

    def _should_display_hover(self, point):
        if False:
            while True:
                i = 10
        'Check if a hover hint should be displayed:'
        if not self._mouse_left_button_pressed:
            return self.hover_hints_enabled and point and self.get_word_at(point)

    def _handle_hover(self):
        if False:
            for i in range(10):
                print('nop')
        'Handle hover hint trigger after delay.'
        self._timer_mouse_moving.stop()
        pos = self._last_point
        ignore_chars = ['(', ')', '.']
        if self._should_display_hover(pos):
            (key, pattern_text, cursor) = self.get_pattern_at(pos)
            text = self.get_word_at(pos)
            if pattern_text:
                ctrl_text = 'Cmd' if sys.platform == 'darwin' else 'Ctrl'
                if key in ['file']:
                    hint_text = ctrl_text + ' + ' + _('click to open file')
                elif key in ['mail']:
                    hint_text = ctrl_text + ' + ' + _('click to send email')
                elif key in ['url']:
                    hint_text = ctrl_text + ' + ' + _('click to open url')
                else:
                    hint_text = ctrl_text + ' + ' + _('click to open')
                hint_text = '<span>&nbsp;{}&nbsp;</span>'.format(hint_text)
                self.show_tooltip(text=hint_text, at_point=pos)
                return
            cursor = self.cursorForPosition(pos)
            cursor_offset = cursor.position()
            (line, col) = (cursor.blockNumber(), cursor.columnNumber())
            self._last_point = pos
            if text and self._last_hover_word != text:
                if all((char not in text for char in ignore_chars)):
                    self._last_hover_word = text
                    self.request_hover(line, col, cursor_offset)
                else:
                    self.hide_tooltip()
        elif not self.is_completion_widget_visible():
            self.hide_tooltip()

    def blockuserdata_list(self):
        if False:
            return 10
        'Get the list of all user data in document.'
        block = self.document().firstBlock()
        while block.isValid():
            data = block.userData()
            if data:
                yield data
            block = block.next()

    def outlineexplorer_data_list(self):
        if False:
            while True:
                i = 10
        'Get the list of all user data in document.'
        for data in self.blockuserdata_list():
            if data.oedata:
                yield data.oedata

    def create_cursor_callback(self, attr):
        if False:
            return 10
        'Make a callback for cursor move event type, (e.g. "Start")'

        def cursor_move_event():
            if False:
                i = 10
                return i + 15
            cursor = self.textCursor()
            move_type = getattr(QTextCursor, attr)
            cursor.movePosition(move_type)
            self.setTextCursor(cursor)
        return cursor_move_event

    def create_shortcuts(self):
        if False:
            i = 10
            return i + 15
        'Create the local shortcuts for the CodeEditor.'
        shortcut_context_name_callbacks = (('editor', 'code completion', self.do_completion), ('editor', 'duplicate line down', self.duplicate_line_down), ('editor', 'duplicate line up', self.duplicate_line_up), ('editor', 'delete line', self.delete_line), ('editor', 'move line up', self.move_line_up), ('editor', 'move line down', self.move_line_down), ('editor', 'go to new line', self.go_to_new_line), ('editor', 'go to definition', self.go_to_definition_from_cursor), ('editor', 'toggle comment', self.toggle_comment), ('editor', 'blockcomment', self.blockcomment), ('editor', 'create_new_cell', self.create_new_cell), ('editor', 'unblockcomment', self.unblockcomment), ('editor', 'transform to uppercase', self.transform_to_uppercase), ('editor', 'transform to lowercase', self.transform_to_lowercase), ('editor', 'indent', lambda : self.indent(force=True)), ('editor', 'unindent', lambda : self.unindent(force=True)), ('editor', 'start of line', self.create_cursor_callback('StartOfLine')), ('editor', 'end of line', self.create_cursor_callback('EndOfLine')), ('editor', 'previous line', self.create_cursor_callback('Up')), ('editor', 'next line', self.create_cursor_callback('Down')), ('editor', 'previous char', self.create_cursor_callback('Left')), ('editor', 'next char', self.create_cursor_callback('Right')), ('editor', 'previous word', self.create_cursor_callback('PreviousWord')), ('editor', 'next word', self.create_cursor_callback('NextWord')), ('editor', 'kill to line end', self.kill_line_end), ('editor', 'kill to line start', self.kill_line_start), ('editor', 'yank', self._kill_ring.yank), ('editor', 'rotate kill ring', self._kill_ring.rotate), ('editor', 'kill previous word', self.kill_prev_word), ('editor', 'kill next word', self.kill_next_word), ('editor', 'start of document', self.create_cursor_callback('Start')), ('editor', 'end of document', self.create_cursor_callback('End')), ('editor', 'undo', self.undo), ('editor', 'redo', self.redo), ('editor', 'cut', self.cut), ('editor', 'copy', self.copy), ('editor', 'paste', self.paste), ('editor', 'delete', self.delete), ('editor', 'select all', self.selectAll), ('editor', 'docstring', self.writer_docstring.write_docstring_for_shortcut), ('editor', 'autoformatting', self.format_document_or_range), ('array_builder', 'enter array inline', self.enter_array_inline), ('array_builder', 'enter array table', self.enter_array_table), ('editor', 'scroll line down', self.scroll_line_down), ('editor', 'scroll line up', self.scroll_line_up))
        shortcuts = []
        for (context, name, callback) in shortcut_context_name_callbacks:
            shortcuts.append(self.config_shortcut(callback, context=context, name=name, parent=self))
        return shortcuts

    def get_shortcut_data(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns shortcut data, a list of tuples (shortcut, text, default)\n        shortcut (QShortcut or QAction instance)\n        text (string): action/shortcut description\n        default (string): default key sequence\n        '
        return [sc.data for sc in self.shortcuts]

    def closeEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.highlighter, sh.PygmentsSH):
            self.highlighter.stop()
        self.update_folding_thread.quit()
        self.update_folding_thread.wait()
        self.update_diagnostics_thread.quit()
        self.update_diagnostics_thread.wait()
        TextEditBaseWidget.closeEvent(self, event)

    def get_document_id(self):
        if False:
            while True:
                i = 10
        return self.document_id

    def set_as_clone(self, editor):
        if False:
            for i in range(10):
                print('nop')
        'Set as clone editor'
        self.setDocument(editor.document())
        self.document_id = editor.get_document_id()
        self.highlighter = editor.highlighter
        self._rehighlight_timer.timeout.connect(self.highlighter.rehighlight)
        self.eol_chars = editor.eol_chars
        self._apply_highlighter_color_scheme()
        self.highlighter.sig_font_changed.connect(self.sync_font)

    def toggle_wrap_mode(self, enable):
        if False:
            for i in range(10):
                print('nop')
        'Enable/disable wrap mode'
        self.set_wrap_mode('word' if enable else None)

    def toggle_line_numbers(self, linenumbers=True, markers=False):
        if False:
            print('Hello World!')
        'Enable/disable line numbers.'
        self.linenumberarea.setup_margins(linenumbers, markers)

    @property
    def panels(self):
        if False:
            return 10
        '\n        Returns a reference to the\n        :class:`spyder.widgets.panels.managers.PanelsManager`\n        used to manage the collection of installed panels\n        '
        return self._panels

    def setup_editor(self, linenumbers=True, language=None, markers=False, font=None, color_scheme=None, wrap=False, tab_mode=True, strip_mode=False, intelligent_backspace=True, automatic_completions=True, automatic_completions_after_chars=3, completions_hint=True, completions_hint_after_ms=500, hover_hints=True, code_snippets=True, highlight_current_line=True, highlight_current_cell=True, occurrence_highlighting=True, scrollflagarea=True, edge_line=True, edge_line_columns=(79,), show_blanks=False, underline_errors=False, close_parentheses=True, close_quotes=False, add_colons=True, auto_unindent=True, indent_chars=' ' * 4, tab_stop_width_spaces=4, cloned_from=None, filename=None, occurrence_timeout=1500, show_class_func_dropdown=False, indent_guides=False, scroll_past_end=False, folding=True, remove_trailing_spaces=False, remove_trailing_newlines=False, add_newline=False, format_on_save=False):
        if False:
            i = 10
            return i + 15
        '\n        Set-up configuration for the CodeEditor instance.\n\n        Usually the parameters here are related with a configurable preference\n        in the Preference Dialog and Editor configurations:\n\n        linenumbers: Enable/Disable line number panel. Default True.\n        language: Set editor language for example python. Default None.\n        markers: Enable/Disable markers panel. Used to show elements like\n            Code Analysis. Default False.\n        font: Base font for the Editor to use. Default None.\n        color_scheme: Initial color scheme for the Editor to use. Default None.\n        wrap: Enable/Disable line wrap. Default False.\n        tab_mode: Enable/Disable using Tab as delimiter between word,\n            Default True.\n        strip_mode: strip_mode: Enable/Disable striping trailing spaces when\n            modifying the file. Default False.\n        intelligent_backspace: Enable/Disable automatically unindenting\n            inserted text (unindenting happens if the leading text length of\n            the line isn\'t module of the length of indentation chars being use)\n            Default True.\n        automatic_completions: Enable/Disable automatic completions.\n            The behavior of the trigger of this the completions can be\n            established with the two following kwargs. Default True.\n        automatic_completions_after_chars: Number of charts to type to trigger\n            an automatic completion. Default 3.\n        completions_hint: Enable/Disable documentation hints for completions.\n            Default True.\n        completions_hint_after_ms: Number of milliseconds over a completion\n            item to show the documentation hint. Default 500.\n        hover_hints: Enable/Disable documentation hover hints. Default True.\n        code_snippets: Enable/Disable code snippets completions. Default True.\n        highlight_current_line: Enable/Disable current line highlighting.\n            Default True.\n        highlight_current_cell: Enable/Disable current cell highlighting.\n            Default True.\n        occurrence_highlighting: Enable/Disable highlighting of current word\n            occurrence in the file. Default True.\n        scrollflagarea : Enable/Disable flag area that shows at the left of\n            the scroll bar. Default True.\n        edge_line: Enable/Disable vertical line to show max number of\n            characters per line. Customizable number of columns in the\n            following kwarg. Default True.\n        edge_line_columns: Number of columns/characters where the editor\n            horizontal edge line will show. Default (79,).\n        show_blanks: Enable/Disable blanks highlighting. Default False.\n        underline_errors: Enable/Disable showing and underline to highlight\n            errors. Default False.\n        close_parentheses: Enable/Disable automatic parentheses closing\n            insertion. Default True.\n        close_quotes: Enable/Disable automatic closing of quotes.\n            Default False.\n        add_colons: Enable/Disable automatic addition of colons. Default True.\n        auto_unindent: Enable/Disable automatically unindentation before else,\n            elif, finally or except statements. Default True.\n        indent_chars: Characters to use for indentation. Default " "*4.\n        tab_stop_width_spaces: Enable/Disable using tabs for indentation.\n            Default 4.\n        cloned_from: Editor instance used as template to instantiate this\n            CodeEditor instance. Default None.\n        filename: Initial filename to show. Default None.\n        occurrence_timeout : Timeout in milliseconds to start highlighting\n            matches/occurrences for the current word under the cursor.\n            Default 1500 ms.\n        show_class_func_dropdown: Enable/Disable a Matlab like widget to show\n            classes and functions available in the current file. Default False.\n        indent_guides: Enable/Disable highlighting of code indentation.\n            Default False.\n        scroll_past_end: Enable/Disable possibility to scroll file passed\n            its end. Default False.\n        folding: Enable/Disable code folding. Default True.\n        remove_trailing_spaces: Remove trailing whitespaces on lines.\n            Default False.\n        remove_trailing_newlines: Remove extra lines at the end of the file.\n            Default False.\n        add_newline: Add a newline at the end of the file if there is not one.\n            Default False.\n        format_on_save: Autoformat file automatically when saving.\n            Default False.\n        '
        self.set_close_parentheses_enabled(close_parentheses)
        self.set_close_quotes_enabled(close_quotes)
        self.set_add_colons_enabled(add_colons)
        self.set_auto_unindent_enabled(auto_unindent)
        self.set_indent_chars(indent_chars)
        self.toggle_code_folding(folding)
        self.scrollflagarea.set_enabled(scrollflagarea)
        self.edge_line.set_enabled(edge_line)
        self.edge_line.set_columns(edge_line_columns)
        self.toggle_identation_guides(indent_guides)
        if self.indent_chars == '\t':
            self.indent_guides.set_indentation_width(tab_stop_width_spaces)
        else:
            self.indent_guides.set_indentation_width(len(self.indent_chars))
        self.set_blanks_enabled(show_blanks)
        self.set_remove_trailing_spaces(remove_trailing_spaces)
        self.set_remove_trailing_newlines(remove_trailing_newlines)
        self.set_add_newline(add_newline)
        self.set_scrollpastend_enabled(scroll_past_end)
        self.toggle_line_numbers(linenumbers, markers)
        self.filename = filename
        self.set_language(language, filename)
        self.set_underline_errors_enabled(underline_errors)
        self.set_highlight_current_cell(highlight_current_cell)
        self.set_highlight_current_line(highlight_current_line)
        self.set_occurrence_highlighting(occurrence_highlighting)
        self.set_occurrence_timeout(occurrence_timeout)
        self.set_tab_mode(tab_mode)
        self.toggle_intelligent_backspace(intelligent_backspace)
        self.toggle_automatic_completions(automatic_completions)
        self.set_automatic_completions_after_chars(automatic_completions_after_chars)
        self.toggle_completions_hint(completions_hint)
        self.set_completions_hint_after_ms(completions_hint_after_ms)
        self.toggle_hover_hints(hover_hints)
        self.toggle_code_snippets(code_snippets)
        self.toggle_format_on_save(format_on_save)
        if cloned_from is not None:
            self.is_cloned = True
            self.setFont(font)
            self.patch = cloned_from.patch
            self.set_as_clone(cloned_from)
            self.panels.refresh()
        elif font is not None:
            self.set_font(font, color_scheme)
        elif color_scheme is not None:
            self.set_color_scheme(color_scheme)
        self.set_tab_stop_width_spaces(tab_stop_width_spaces)
        self.toggle_wrap_mode(wrap)
        self.classfuncdropdown.setVisible(show_class_func_dropdown and self.is_python_like())
        self.set_strip_mode(strip_mode)

    def set_folding_panel(self, folding):
        if False:
            i = 10
            return i + 15
        'Enable/disable folding panel.'
        folding_panel = self.panels.get(FoldingPanel)
        folding_panel.setVisible(folding)

    def set_tab_mode(self, enable):
        if False:
            while True:
                i = 10
        '\n        enabled = tab always indent\n        (otherwise tab indents only when cursor is at the beginning of a line)\n        '
        self.tab_mode = enable

    def set_strip_mode(self, enable):
        if False:
            i = 10
            return i + 15
        '\n        Strip all trailing spaces if enabled.\n        '
        self.strip_trailing_spaces_on_modify = enable

    def toggle_intelligent_backspace(self, state):
        if False:
            print('Hello World!')
        self.intelligent_backspace = state

    def toggle_automatic_completions(self, state):
        if False:
            i = 10
            return i + 15
        self.automatic_completions = state

    def toggle_hover_hints(self, state):
        if False:
            print('Hello World!')
        self.hover_hints_enabled = state

    def toggle_code_snippets(self, state):
        if False:
            while True:
                i = 10
        self.code_snippets = state

    def toggle_format_on_save(self, state):
        if False:
            while True:
                i = 10
        self.format_on_save = state

    def toggle_code_folding(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.code_folding = state
        self.set_folding_panel(state)
        if not state and self.indent_guides._enabled:
            self.code_folding = True

    def toggle_identation_guides(self, state):
        if False:
            return 10
        if state and (not self.code_folding):
            self.code_folding = True
        self.indent_guides.set_enabled(state)

    def toggle_completions_hint(self, state):
        if False:
            for i in range(10):
                print('nop')
        'Enable/disable completion hint.'
        self.completions_hint = state

    def set_automatic_completions_after_chars(self, number):
        if False:
            while True:
                i = 10
        '\n        Set the number of characters after which auto completion is fired.\n        '
        self.automatic_completions_after_chars = number

    def set_completions_hint_after_ms(self, ms):
        if False:
            return 10
        '\n        Set the amount of time in ms after which the completions hint is shown.\n        '
        self.completions_hint_after_ms = ms

    def set_close_parentheses_enabled(self, enable):
        if False:
            print('Hello World!')
        'Enable/disable automatic parentheses insertion feature'
        self.close_parentheses_enabled = enable
        bracket_extension = self.editor_extensions.get(CloseBracketsExtension)
        if bracket_extension is not None:
            bracket_extension.enabled = enable

    def set_close_quotes_enabled(self, enable):
        if False:
            i = 10
            return i + 15
        'Enable/disable automatic quote insertion feature'
        self.close_quotes_enabled = enable
        quote_extension = self.editor_extensions.get(CloseQuotesExtension)
        if quote_extension is not None:
            quote_extension.enabled = enable

    def set_add_colons_enabled(self, enable):
        if False:
            while True:
                i = 10
        'Enable/disable automatic colons insertion feature'
        self.add_colons_enabled = enable

    def set_auto_unindent_enabled(self, enable):
        if False:
            for i in range(10):
                print('nop')
        'Enable/disable automatic unindent after else/elif/finally/except'
        self.auto_unindent_enabled = enable

    def set_occurrence_highlighting(self, enable):
        if False:
            for i in range(10):
                print('nop')
        'Enable/disable occurrence highlighting'
        self.occurrence_highlighting = enable
        if not enable:
            self.clear_occurrences()

    def set_occurrence_timeout(self, timeout):
        if False:
            i = 10
            return i + 15
        'Set occurrence highlighting timeout (ms)'
        self.occurrence_timer.setInterval(timeout)

    def set_underline_errors_enabled(self, state):
        if False:
            while True:
                i = 10
        'Toggle the underlining of errors and warnings.'
        self.underline_errors_enabled = state
        if not state:
            self.clear_extra_selections('code_analysis_underline')

    def set_highlight_current_line(self, enable):
        if False:
            return 10
        'Enable/disable current line highlighting'
        self.highlight_current_line_enabled = enable
        if self.highlight_current_line_enabled:
            self.highlight_current_line()
        else:
            self.unhighlight_current_line()

    def set_highlight_current_cell(self, enable):
        if False:
            i = 10
            return i + 15
        'Enable/disable current line highlighting'
        hl_cell_enable = enable and self.supported_cell_language
        self.highlight_current_cell_enabled = hl_cell_enable
        if self.highlight_current_cell_enabled:
            self.highlight_current_cell()
        else:
            self.unhighlight_current_cell()

    def set_language(self, language, filename=None):
        if False:
            print('Hello World!')
        extra_supported_languages = {'stil': 'STIL'}
        self.tab_indents = language in self.TAB_ALWAYS_INDENTS
        self.comment_string = ''
        self.language = 'Text'
        self.supported_language = False
        sh_class = sh.TextSH
        language = 'None' if language is None else language
        if language is not None:
            for (key, value) in ALL_LANGUAGES.items():
                if language.lower() in value:
                    self.supported_language = True
                    (sh_class, comment_string) = self.LANGUAGES[key]
                    if key == 'IPython':
                        self.language = 'Python'
                    else:
                        self.language = key
                    self.comment_string = comment_string
                    if key in CELL_LANGUAGES:
                        self.supported_cell_language = True
                        self.has_cell_separators = True
                    break
        if filename is not None and (not self.supported_language):
            sh_class = sh.guess_pygments_highlighter(filename)
            self.support_language = sh_class is not sh.TextSH
            if self.support_language:
                if sh_class._lexer.name == 'S':
                    self.language = 'R'
                else:
                    self.language = sh_class._lexer.name
            else:
                (_, ext) = osp.splitext(filename)
                ext = ext.lower()
                if ext in extra_supported_languages:
                    self.language = extra_supported_languages[ext]
        self._set_highlighter(sh_class)
        self.completion_widget.set_language(self.language)

    def _set_highlighter(self, sh_class):
        if False:
            return 10
        self.highlighter_class = sh_class
        if self.highlighter is not None:
            self.highlighter.setParent(None)
            self.highlighter.setDocument(None)
        self.highlighter = self.highlighter_class(self.document(), self.font(), self.color_scheme)
        self._apply_highlighter_color_scheme()
        self.highlighter.editor = self
        self.highlighter.sig_font_changed.connect(self.sync_font)
        self._rehighlight_timer.timeout.connect(self.highlighter.rehighlight)

    def sync_font(self):
        if False:
            print('Hello World!')
        'Highlighter changed font, update.'
        self.setFont(self.highlighter.font)
        self.sig_font_changed.emit()

    def get_cell_list(self):
        if False:
            while True:
                i = 10
        'Get all cells.'
        if self.highlighter is None:
            return []

        def good(oedata):
            if False:
                while True:
                    i = 10
            return oedata.is_valid() and oedata.def_type == oedata.CELL
        self.highlighter._cell_list = [oedata for oedata in self.highlighter._cell_list if good(oedata)]
        return sorted({oedata.block.blockNumber(): oedata for oedata in self.highlighter._cell_list}.items())

    def is_json(self):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(self.highlighter, sh.PygmentsSH) and self.highlighter._lexer.name == 'JSON'

    def is_python(self):
        if False:
            while True:
                i = 10
        return self.highlighter_class is sh.PythonSH

    def is_ipython(self):
        if False:
            while True:
                i = 10
        return self.highlighter_class is sh.IPythonSH

    def is_python_or_ipython(self):
        if False:
            for i in range(10):
                print('nop')
        return self.is_python() or self.is_ipython()

    def is_cython(self):
        if False:
            return 10
        return self.highlighter_class is sh.CythonSH

    def is_enaml(self):
        if False:
            return 10
        return self.highlighter_class is sh.EnamlSH

    def is_python_like(self):
        if False:
            for i in range(10):
                print('nop')
        return self.is_python() or self.is_ipython() or self.is_cython() or self.is_enaml()

    def intelligent_tab(self):
        if False:
            print('Hello World!')
        'Provide intelligent behavior for Tab key press.'
        leading_text = self.get_text('sol', 'cursor')
        if not leading_text.strip() or leading_text.endswith('#'):
            self.indent_or_replace()
        elif self.in_comment_or_string() and (not leading_text.endswith(' ')):
            self.do_completion()
        elif leading_text.endswith('import ') or leading_text[-1] == '.':
            self.do_completion()
        elif leading_text.split()[0] in ['from', 'import'] and ';' not in leading_text:
            self.do_completion()
        elif leading_text[-1] in '(,' or leading_text.endswith(', '):
            self.indent_or_replace()
        elif leading_text.endswith(' '):
            self.indent_or_replace()
        elif re.search('[^\\d\\W]\\w*\\Z', leading_text, re.UNICODE):
            self.do_completion()
        else:
            self.indent_or_replace()

    def intelligent_backtab(self):
        if False:
            i = 10
            return i + 15
        'Provide intelligent behavior for Shift+Tab key press'
        leading_text = self.get_text('sol', 'cursor')
        if not leading_text.strip():
            self.unindent()
        elif self.in_comment_or_string():
            self.unindent()
        elif leading_text[-1] in '(,' or leading_text.endswith(', '):
            self.show_object_info()
        else:
            self.unindent()

    def rehighlight(self):
        if False:
            print('Hello World!')
        'Rehighlight the whole document.'
        if self.highlighter is not None:
            self.highlighter.rehighlight()
        if self.highlight_current_cell_enabled:
            self.highlight_current_cell()
        else:
            self.unhighlight_current_cell()
        if self.highlight_current_line_enabled:
            self.highlight_current_line()
        else:
            self.unhighlight_current_line()

    def trim_trailing_spaces(self):
        if False:
            i = 10
            return i + 15
        'Remove trailing spaces'
        cursor = self.textCursor()
        cursor.beginEditBlock()
        cursor.movePosition(QTextCursor.Start)
        while True:
            cursor.movePosition(QTextCursor.EndOfBlock)
            text = to_text_string(cursor.block().text())
            length = len(text) - len(text.rstrip())
            if length > 0:
                cursor.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, length)
                cursor.removeSelectedText()
            if cursor.atEnd():
                break
            cursor.movePosition(QTextCursor.NextBlock)
        cursor.endEditBlock()

    def trim_trailing_newlines(self):
        if False:
            while True:
                i = 10
        'Remove extra newlines at the end of the document.'
        cursor = self.textCursor()
        cursor.beginEditBlock()
        cursor.movePosition(QTextCursor.End)
        line = cursor.blockNumber()
        this_line = self.get_text_line(line)
        previous_line = self.get_text_line(line - 1)
        if self.get_line_count() > 1:
            while this_line == '':
                cursor.movePosition(QTextCursor.PreviousBlock, QTextCursor.KeepAnchor)
                if self.add_newline:
                    if this_line == '' and previous_line != '':
                        cursor.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor)
                line -= 1
                if line == 0:
                    break
                this_line = self.get_text_line(line)
                previous_line = self.get_text_line(line - 1)
        if not self.add_newline:
            cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.endEditBlock()

    def add_newline_to_file(self):
        if False:
            return 10
        'Add a newline to the end of the file if it does not exist.'
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        line = cursor.blockNumber()
        this_line = self.get_text_line(line)
        if this_line != '':
            cursor.beginEditBlock()
            cursor.movePosition(QTextCursor.EndOfBlock)
            cursor.insertText(self.get_line_separator())
            cursor.endEditBlock()

    def fix_indentation(self):
        if False:
            print('Hello World!')
        'Replace tabs by spaces.'
        text_before = to_text_string(self.toPlainText())
        text_after = sourcecode.fix_indentation(text_before, self.indent_chars)
        if text_before != text_after:
            self.selectAll()
            self.skip_rstrip = True
            self.insertPlainText(text_after)
            self.skip_rstrip = False

    def get_current_object(self):
        if False:
            return 10
        'Return current object (string) '
        source_code = to_text_string(self.toPlainText())
        offset = self.get_position('cursor')
        return sourcecode.get_primary_at(source_code, offset)

    def next_cursor_position(self, position=None, mode=QTextLayout.SkipCharacters):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get next valid cursor position.\n\n        Adapted from:\n        https://github.com/qt/qtbase/blob/5.15.2/src/gui/text/qtextdocument_p.cpp#L1361\n        '
        cursor = self.textCursor()
        if cursor.atEnd():
            return position
        if position is None:
            position = cursor.position()
        else:
            cursor.setPosition(position)
        it = cursor.block()
        start = it.position()
        end = start + it.length() - 1
        if position == end:
            return end + 1
        return it.layout().nextCursorPosition(position - start, mode) + start

    @Slot()
    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        'Remove selected text or next character.'
        if not self.has_selected_text():
            cursor = self.textCursor()
            if not cursor.atEnd():
                cursor.setPosition(self.next_cursor_position(), QTextCursor.KeepAnchor)
            self.setTextCursor(cursor)
        self.remove_selected_text()

    def scroll_line_down(self):
        if False:
            print('Hello World!')
        'Scroll the editor down by one step.'
        vsb = self.verticalScrollBar()
        vsb.setValue(vsb.value() + vsb.singleStep())

    def scroll_line_up(self):
        if False:
            return 10
        'Scroll the editor up by one step.'
        vsb = self.verticalScrollBar()
        vsb.setValue(vsb.value() - vsb.singleStep())

    def __find_first(self, text):
        if False:
            i = 10
            return i + 15
        'Find first occurrence: scan whole document'
        flags = QTextDocument.FindCaseSensitively | QTextDocument.FindWholeWords
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Start)
        regexp = QRegExp('\\b%s\\b' % QRegExp.escape(text), Qt.CaseSensitive)
        cursor = self.document().find(regexp, cursor, flags)
        self.__find_first_pos = cursor.position()
        return cursor

    def __find_next(self, text, cursor):
        if False:
            return 10
        'Find next occurrence'
        flags = QTextDocument.FindCaseSensitively | QTextDocument.FindWholeWords
        regexp = QRegExp('\\b%s\\b' % QRegExp.escape(text), Qt.CaseSensitive)
        cursor = self.document().find(regexp, cursor, flags)
        if cursor.position() != self.__find_first_pos:
            return cursor

    def __cursor_position_changed(self):
        if False:
            for i in range(10):
                print('nop')
        'Cursor position has changed'
        (line, column) = self.get_cursor_line_column()
        self.sig_cursor_position_changed.emit(line, column)
        if self.highlight_current_cell_enabled:
            self.highlight_current_cell()
        else:
            self.unhighlight_current_cell()
        if self.highlight_current_line_enabled:
            self.highlight_current_line()
        else:
            self.unhighlight_current_line()
        if self.occurrence_highlighting:
            self.occurrence_timer.start()
        self.strip_trailing_spaces()

    def clear_occurrences(self):
        if False:
            while True:
                i = 10
        'Clear occurrence markers'
        self.occurrences = []
        self.clear_extra_selections('occurrences')
        self.sig_flags_changed.emit()

    def get_selection(self, cursor, foreground_color=None, background_color=None, underline_color=None, outline_color=None, underline_style=QTextCharFormat.SingleUnderline):
        if False:
            while True:
                i = 10
        'Get selection.'
        if cursor is None:
            return
        selection = TextDecoration(cursor)
        if foreground_color is not None:
            selection.format.setForeground(foreground_color)
        if background_color is not None:
            selection.format.setBackground(background_color)
        if underline_color is not None:
            selection.format.setProperty(QTextFormat.TextUnderlineStyle, to_qvariant(underline_style))
            selection.format.setProperty(QTextFormat.TextUnderlineColor, to_qvariant(underline_color))
        if outline_color is not None:
            selection.set_outline(outline_color)
        return selection

    def highlight_selection(self, key, cursor, foreground_color=None, background_color=None, underline_color=None, outline_color=None, underline_style=QTextCharFormat.SingleUnderline):
        if False:
            i = 10
            return i + 15
        selection = self.get_selection(cursor, foreground_color, background_color, underline_color, outline_color, underline_style)
        if selection is None:
            return
        extra_selections = self.get_extra_selections(key)
        extra_selections.append(selection)
        self.set_extra_selections(key, extra_selections)

    def mark_occurrences(self):
        if False:
            for i in range(10):
                print('nop')
        'Marking occurrences of the currently selected word'
        self.clear_occurrences()
        if not self.supported_language:
            return
        text = self.get_selected_text().strip()
        if not text:
            text = self.get_current_word()
        if text is None:
            return
        if self.has_selected_text() and self.get_selected_text().strip() != text:
            return
        if self.is_python_like() and (sourcecode.is_keyword(to_text_string(text)) or to_text_string(text) == 'self'):
            return
        cursor = self.__find_first(text)
        self.occurrences = []
        extra_selections = self.get_extra_selections('occurrences')
        first_occurrence = None
        while cursor:
            block = cursor.block()
            if not block.userData():
                block.setUserData(BlockUserData(self))
            self.occurrences.append(block)
            selection = self.get_selection(cursor)
            if len(selection.cursor.selectedText()) > 0:
                extra_selections.append(selection)
                if len(extra_selections) == 1:
                    first_occurrence = selection
                else:
                    selection.format.setBackground(self.occurrence_color)
                    first_occurrence.format.setBackground(self.occurrence_color)
            cursor = self.__find_next(text, cursor)
        self.set_extra_selections('occurrences', extra_selections)
        if len(self.occurrences) > 1 and self.occurrences[-1] == 0:
            self.occurrences.pop(-1)
        self.sig_flags_changed.emit()

    def highlight_found_results(self, pattern, word=False, regexp=False, case=False):
        if False:
            for i in range(10):
                print('nop')
        'Highlight all found patterns'
        self.__find_args = {'pattern': pattern, 'word': word, 'regexp': regexp, 'case': case}
        pattern = to_text_string(pattern)
        if not pattern:
            return
        if not regexp:
            pattern = re.escape(to_text_string(pattern))
        pattern = '\\b%s\\b' % pattern if word else pattern
        text = to_text_string(self.toPlainText())
        re_flags = re.MULTILINE if case else re.IGNORECASE | re.MULTILINE
        try:
            regobj = re.compile(pattern, flags=re_flags)
        except sre_constants.error:
            return
        extra_selections = []
        self.found_results = []
        has_unicode = len(text) != qstring_length(text)
        for match in regobj.finditer(text):
            if has_unicode:
                (pos1, pos2) = sh.get_span(match)
            else:
                (pos1, pos2) = match.span()
            selection = TextDecoration(self.textCursor())
            selection.format.setBackground(self.found_results_color)
            selection.cursor.setPosition(pos1)
            block = selection.cursor.block()
            if not block.userData():
                block.setUserData(BlockUserData(self))
            self.found_results.append(block)
            selection.cursor.setPosition(pos2, QTextCursor.KeepAnchor)
            extra_selections.append(selection)
        self.set_extra_selections('find', extra_selections)

    def clear_found_results(self):
        if False:
            print('Hello World!')
        'Clear found results highlighting'
        self.found_results = []
        self.clear_extra_selections('find')
        self.sig_flags_changed.emit()

    def __text_has_changed(self):
        if False:
            print('Hello World!')
        'Text has changed, eventually clear found results highlighting'
        self.last_change_position = self.textCursor().position()
        for result in self.found_results:
            self.highlight_found_results(**self.__find_args)
            break

    def get_linenumberarea_width(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return current line number area width.\n\n        This method is left for backward compatibility (BaseEditMixin\n        define it), any changes should be in LineNumberArea class.\n        '
        return self.linenumberarea.get_width()

    def calculate_real_position(self, point):
        if False:
            i = 10
            return i + 15
        'Add offset to a point, to take into account the panels.'
        point.setX(point.x() + self.panels.margin_size(Panel.Position.LEFT))
        point.setY(point.y() + self.panels.margin_size(Panel.Position.TOP))
        return point

    def calculate_real_position_from_global(self, point):
        if False:
            i = 10
            return i + 15
        'Add offset to a point, to take into account the panels.'
        point.setX(point.x() - self.panels.margin_size(Panel.Position.LEFT))
        point.setY(point.y() + self.panels.margin_size(Panel.Position.TOP))
        return point

    def get_linenumber_from_mouse_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Return line number from mouse event'
        block = self.firstVisibleBlock()
        line_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()
        while block.isValid() and top < event.pos().y():
            block = block.next()
            if block.isVisible():
                top = bottom
                bottom = top + self.blockBoundingRect(block).height()
            line_number += 1
        return line_number

    def select_lines(self, linenumber_pressed, linenumber_released):
        if False:
            for i in range(10):
                print('nop')
        'Select line(s) after a mouse press/mouse press drag event'
        find_block_by_number = self.document().findBlockByNumber
        move_n_blocks = linenumber_released - linenumber_pressed
        start_line = linenumber_pressed
        start_block = find_block_by_number(start_line - 1)
        cursor = self.textCursor()
        cursor.setPosition(start_block.position())
        if move_n_blocks > 0:
            for n in range(abs(move_n_blocks) + 1):
                cursor.movePosition(cursor.NextBlock, cursor.KeepAnchor)
        else:
            cursor.movePosition(cursor.NextBlock)
            for n in range(abs(move_n_blocks) + 1):
                cursor.movePosition(cursor.PreviousBlock, cursor.KeepAnchor)
        if linenumber_released == self.blockCount():
            cursor.movePosition(cursor.EndOfBlock, cursor.KeepAnchor)
        else:
            cursor.movePosition(cursor.StartOfBlock, cursor.KeepAnchor)
        self.setTextCursor(cursor)

    def add_bookmark(self, slot_num, line=None, column=None):
        if False:
            print('Hello World!')
        "Add bookmark to current block's userData."
        if line is None:
            (line, column) = self.get_cursor_line_column()
        block = self.document().findBlockByNumber(line)
        data = block.userData()
        if not data:
            data = BlockUserData(self)
        if slot_num not in data.bookmarks:
            data.bookmarks.append((slot_num, column))
        block.setUserData(data)
        self._bookmarks_blocks[id(block)] = block
        self.sig_bookmarks_changed.emit()

    def get_bookmarks(self):
        if False:
            for i in range(10):
                print('nop')
        'Get bookmarks by going over all blocks.'
        bookmarks = {}
        pruned_bookmarks_blocks = {}
        for block_id in self._bookmarks_blocks:
            block = self._bookmarks_blocks[block_id]
            if block.isValid():
                data = block.userData()
                if data and data.bookmarks:
                    pruned_bookmarks_blocks[block_id] = block
                    line_number = block.blockNumber()
                    for (slot_num, column) in data.bookmarks:
                        bookmarks[slot_num] = [line_number, column]
        self._bookmarks_blocks = pruned_bookmarks_blocks
        return bookmarks

    def clear_bookmarks(self):
        if False:
            return 10
        'Clear bookmarks for all blocks.'
        self.bookmarks = {}
        for data in self.blockuserdata_list():
            data.bookmarks = []
        self._bookmarks_blocks = {}

    def set_bookmarks(self, bookmarks):
        if False:
            print('Hello World!')
        'Set bookmarks when opening file.'
        self.clear_bookmarks()
        for (slot_num, bookmark) in bookmarks.items():
            self.add_bookmark(slot_num, bookmark[1], bookmark[2])

    def update_bookmarks(self):
        if False:
            while True:
                i = 10
        'Emit signal to update bookmarks.'
        self.sig_bookmarks_changed.emit()

    def show_completion_object_info(self, name, signature):
        if False:
            print('Hello World!')
        'Trigger show completion info in Help Pane.'
        force = True
        self.sig_show_completion_object_info.emit(name, signature, force)

    @Slot()
    def show_object_info(self):
        if False:
            print('Hello World!')
        'Trigger a calltip'
        self.sig_show_object_info.emit(True)

    def set_blanks_enabled(self, state):
        if False:
            print('Hello World!')
        'Toggle blanks visibility'
        self.blanks_enabled = state
        option = self.document().defaultTextOption()
        option.setFlags(option.flags() | QTextOption.AddSpaceForLineAndParagraphSeparators)
        if self.blanks_enabled:
            option.setFlags(option.flags() | QTextOption.ShowTabsAndSpaces)
        else:
            option.setFlags(option.flags() & ~QTextOption.ShowTabsAndSpaces)
        self.document().setDefaultTextOption(option)
        self.rehighlight()

    def set_scrollpastend_enabled(self, state):
        if False:
            return 10
        '\n        Allow user to scroll past the end of the document to have the last\n        line on top of the screen\n        '
        self.scrollpastend_enabled = state
        self.setCenterOnScroll(state)
        self.setDocument(self.document())

    def resizeEvent(self, event):
        if False:
            return 10
        'Reimplemented Qt method to handle p resizing'
        TextEditBaseWidget.resizeEvent(self, event)
        self.panels.resize()

    def showEvent(self, event):
        if False:
            return 10
        'Overrides showEvent to update the viewport margins.'
        super(CodeEditor, self).showEvent(event)
        self.panels.refresh()

    def _apply_highlighter_color_scheme(self):
        if False:
            return 10
        'Apply color scheme from syntax highlighter to the editor'
        hl = self.highlighter
        if hl is not None:
            self.set_palette(background=hl.get_background_color(), foreground=hl.get_foreground_color())
            self.currentline_color = hl.get_currentline_color()
            self.currentcell_color = hl.get_currentcell_color()
            self.occurrence_color = hl.get_occurrence_color()
            self.ctrl_click_color = hl.get_ctrlclick_color()
            self.sideareas_color = hl.get_sideareas_color()
            self.comment_color = hl.get_comment_color()
            self.normal_color = hl.get_foreground_color()
            self.matched_p_color = hl.get_matched_p_color()
            self.unmatched_p_color = hl.get_unmatched_p_color()
            self.edge_line.update_color()
            self.indent_guides.update_color()
            self.sig_theme_colors_changed.emit({'occurrence': self.occurrence_color})

    def apply_highlighter_settings(self, color_scheme=None):
        if False:
            for i in range(10):
                print('nop')
        'Apply syntax highlighter settings'
        if self.highlighter is not None:
            self.highlighter.setup_formats(self.font())
            if color_scheme is not None:
                self.set_color_scheme(color_scheme)
            else:
                self._rehighlight_timer.start()

    def set_font(self, font, color_scheme=None):
        if False:
            print('Hello World!')
        'Set font'
        if color_scheme is not None:
            self.color_scheme = color_scheme
        self.setFont(font)
        self.panels.refresh()
        self.apply_highlighter_settings(color_scheme)

    def set_color_scheme(self, color_scheme):
        if False:
            return 10
        'Set color scheme for syntax highlighting'
        self.color_scheme = color_scheme
        if self.highlighter is not None:
            self.highlighter.set_color_scheme(color_scheme)
            self._apply_highlighter_color_scheme()
        if self.highlight_current_cell_enabled:
            self.highlight_current_cell()
        else:
            self.unhighlight_current_cell()
        if self.highlight_current_line_enabled:
            self.highlight_current_line()
        else:
            self.unhighlight_current_line()

    def set_text(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Set the text of the editor'
        self.setPlainText(text)
        self.set_eol_chars(text=text)
        if isinstance(self.highlighter, sh.PygmentsSH) and (not running_under_pytest()):
            self.highlighter.make_charlist()

    def set_text_from_file(self, filename, language=None):
        if False:
            for i in range(10):
                print('nop')
        'Set the text of the editor from file *fname*'
        self.filename = filename
        (text, _enc) = encoding.read(filename)
        if language is None:
            language = get_file_language(filename, text)
        self.set_language(language, filename)
        self.set_text(text)

    def append(self, text):
        if False:
            print('Hello World!')
        'Append text to the end of the text widget'
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)

    def adjust_indentation(self, line, indent_adjustment):
        if False:
            i = 10
            return i + 15
        'Adjust indentation.'
        if indent_adjustment == 0 or line == '':
            return line
        using_spaces = self.indent_chars != '\t'
        if indent_adjustment > 0:
            if using_spaces:
                return ' ' * indent_adjustment + line
            else:
                return self.indent_chars * (indent_adjustment // self.tab_stop_width_spaces) + line
        max_indent = self.get_line_indentation(line)
        indent_adjustment = min(max_indent, -indent_adjustment)
        indent_adjustment = indent_adjustment if using_spaces else indent_adjustment // self.tab_stop_width_spaces
        return line[indent_adjustment:]

    @Slot()
    def paste(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Insert text or file/folder path copied from clipboard.\n\n        Reimplement QPlainTextEdit's method to fix the following issue:\n        on Windows, pasted text has only 'LF' EOL chars even if the original\n        text has 'CRLF' EOL chars.\n        The function also changes the clipboard data if they are copied as\n        files/folders but does not change normal text data except if they are\n        multiple lines. Since we are changing clipboard data we cannot use\n        paste, which directly pastes from clipboard instead we use\n        insertPlainText and pass the formatted/changed text without modifying\n        clipboard content.\n        "
        clipboard = QApplication.clipboard()
        text = to_text_string(clipboard.text())
        if clipboard.mimeData().hasUrls():
            urls = clipboard.mimeData().urls()
            if all([url.isLocalFile() for url in urls]):
                if len(urls) > 1:
                    sep_chars = ',' + self.get_line_separator()
                    text = sep_chars.join(('"' + url.toLocalFile().replace(osp.os.sep, '/') + '"' for url in urls))
                elif urls:
                    text = urls[0].toLocalFile().replace(osp.os.sep, '/')
        eol_chars = self.get_line_separator()
        if len(text.splitlines()) > 1:
            text = eol_chars.join((text + eol_chars).splitlines())
        cursor = self.textCursor()
        cursor.beginEditBlock()
        cursor.removeSelectedText()
        cursor.setPosition(cursor.selectionStart())
        cursor.setPosition(cursor.block().position(), QTextCursor.KeepAnchor)
        preceding_text = cursor.selectedText()
        (first_line_selected, *remaining_lines) = (text + eol_chars).splitlines()
        first_line = preceding_text + first_line_selected
        first_line_adjustment = 0
        if self.is_python_like() and len(preceding_text.strip()) == 0 and (len(first_line.strip()) > 0):
            desired_indent = self.find_indentation()
            if desired_indent:
                desired_indent = max(desired_indent, self.get_line_indentation(first_line_selected), self.get_line_indentation(preceding_text))
                first_line_adjustment = desired_indent - self.get_line_indentation(first_line)
                first_line_adjustment = min(first_line_adjustment, 0)
                first_line = self.adjust_indentation(first_line, first_line_adjustment)
        if len(remaining_lines) > 0 and len(first_line.strip()) > 0:
            lines_adjustment = first_line_adjustment
            lines_adjustment += CLIPBOARD_HELPER.remaining_lines_adjustment(preceding_text)
            indentations = [self.get_line_indentation(line) for line in remaining_lines if line.strip() != '']
            if indentations:
                max_dedent = min(indentations)
                lines_adjustment = max(lines_adjustment, -max_dedent)
            remaining_lines = [self.adjust_indentation(line, lines_adjustment) for line in remaining_lines]
        text = eol_chars.join([first_line, *remaining_lines])
        self.skip_rstrip = True
        self.sig_will_paste_text.emit(text)
        cursor.removeSelectedText()
        cursor.insertText(text)
        cursor.endEditBlock()
        self.sig_text_was_inserted.emit()
        self.skip_rstrip = False

    def _save_clipboard_indentation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Save the indentation corresponding to the clipboard data.\n\n        Must be called right after copying.\n        '
        cursor = self.textCursor()
        cursor.setPosition(cursor.selectionStart())
        cursor.setPosition(cursor.block().position(), QTextCursor.KeepAnchor)
        preceding_text = cursor.selectedText()
        CLIPBOARD_HELPER.save_indentation(preceding_text, self.tab_stop_width_spaces)

    @Slot()
    def cut(self):
        if False:
            for i in range(10):
                print('nop')
        'Reimplement cut to signal listeners about changes on the text.'
        has_selected_text = self.has_selected_text()
        if not has_selected_text:
            return
        (start, end) = self.get_selection_start_end()
        self.sig_will_remove_selection.emit(start, end)
        TextEditBaseWidget.cut(self)
        self._save_clipboard_indentation()
        self.sig_text_was_inserted.emit()

    @Slot()
    def copy(self):
        if False:
            while True:
                i = 10
        'Reimplement copy to save indentation.'
        TextEditBaseWidget.copy(self)
        self._save_clipboard_indentation()

    @Slot()
    def undo(self):
        if False:
            print('Hello World!')
        'Reimplement undo to decrease text version number.'
        if self.document().isUndoAvailable():
            self.text_version -= 1
            self.skip_rstrip = True
            self.is_undoing = True
            TextEditBaseWidget.undo(self)
            self.sig_undo.emit()
            self.sig_text_was_inserted.emit()
            self.is_undoing = False
            self.skip_rstrip = False

    @Slot()
    def redo(self):
        if False:
            return 10
        'Reimplement redo to increase text version number.'
        if self.document().isRedoAvailable():
            self.text_version += 1
            self.skip_rstrip = True
            self.is_redoing = True
            TextEditBaseWidget.redo(self)
            self.sig_redo.emit()
            self.sig_text_was_inserted.emit()
            self.is_redoing = False
            self.skip_rstrip = False

    @Slot()
    def center_cursor_on_next_focus(self):
        if False:
            i = 10
            return i + 15
        'QPlainTextEdit\'s "centerCursor" requires the widget to be visible'
        self.centerCursor()
        self.focus_in.disconnect(self.center_cursor_on_next_focus)

    def go_to_line(self, line, start_column=0, end_column=0, word=''):
        if False:
            for i in range(10):
                print('nop')
        'Go to line number *line* and eventually highlight it'
        self.text_helper.goto_line(line, column=start_column, end_column=end_column, move=True, word=word)

    def exec_gotolinedialog(self):
        if False:
            print('Hello World!')
        'Execute the GoToLineDialog dialog box'
        dlg = GoToLineDialog(self)
        if dlg.exec_():
            self.go_to_line(dlg.get_line_number())

    def hide_tooltip(self):
        if False:
            i = 10
            return i + 15
        '\n        Hide the tooltip widget.\n\n        The tooltip widget is a special QLabel that looks like a tooltip,\n        this method is here so it can be hidden as necessary. For example,\n        when the user leaves the Linenumber area when hovering over lint\n        warnings and errors.\n        '
        self._timer_mouse_moving.stop()
        self._last_hover_word = None
        self.clear_extra_selections('code_analysis_highlight')
        if self.tooltip_widget.isVisible():
            self.tooltip_widget.hide()

    def _set_completions_hint_idle(self):
        if False:
            return 10
        self._completions_hint_idle = True
        self.completion_widget.trigger_completion_hint()

    def show_hint_for_completion(self, word, documentation, at_point):
        if False:
            print('Hello World!')
        'Show hint for completion element.'
        if self.completions_hint and self._completions_hint_idle:
            documentation = documentation.replace(u'\xa0', ' ')
            completion_doc = {'name': word, 'signature': documentation}
            if documentation and len(documentation) > 0:
                self.show_hint(documentation, inspect_word=word, at_point=at_point, completion_doc=completion_doc, max_lines=self._DEFAULT_MAX_LINES, max_width=self._DEFAULT_COMPLETION_HINT_MAX_WIDTH)
                self.tooltip_widget.move(at_point)
                return
        self.hide_tooltip()

    def update_decorations(self):
        if False:
            print('Hello World!')
        'Update decorations on the visible portion of the screen.'
        if self.underline_errors_enabled:
            self.underline_errors()
        self.decorations.update()

    def show_code_analysis_results(self, line_number, block_data):
        if False:
            for i in range(10):
                print('nop')
        'Show warning/error messages.'
        icons = {DiagnosticSeverity.ERROR: 'error', DiagnosticSeverity.WARNING: 'warning', DiagnosticSeverity.INFORMATION: 'information', DiagnosticSeverity.HINT: 'hint'}
        code_analysis = block_data.code_analysis
        fm = self.fontMetrics()
        size = fm.height()
        template = '<img src="data:image/png;base64, {}" height="{size}" width="{size}" />&nbsp;{} <i>({} {})</i>'
        msglist = []
        max_lines_msglist = 25
        sorted_code_analysis = sorted(code_analysis, key=lambda i: i[2])
        for (src, code, sev, msg) in sorted_code_analysis:
            if src == 'pylint' and '[' in msg and (']' in msg):
                msg = msg.split(']')[-1]
            msg = msg.strip()
            if len(msg) > 1:
                msg = msg[0].upper() + msg[1:]
            msg = msg.replace('<', '&lt;').replace('>', '&gt;')
            paragraphs = msg.splitlines()
            new_paragraphs = []
            long_paragraphs = 0
            lines_per_message = 6
            for paragraph in paragraphs:
                new_paragraph = textwrap.wrap(paragraph, width=self._DEFAULT_MAX_HINT_WIDTH)
                if lines_per_message > 2:
                    if len(new_paragraph) > 1:
                        new_paragraph = '<br>'.join(new_paragraph[:2]) + '...'
                        long_paragraphs += 1
                        lines_per_message -= 2
                    else:
                        new_paragraph = '<br>'.join(new_paragraph)
                        lines_per_message -= 1
                    new_paragraphs.append(new_paragraph)
            if len(new_paragraphs) > 1:
                if long_paragraphs != 0:
                    max_lines = 3
                    max_lines_msglist -= max_lines * 2
                else:
                    max_lines = 5
                    max_lines_msglist -= max_lines
                msg = '<br>'.join(new_paragraphs[:max_lines]) + '<br>'
            else:
                msg = '<br>'.join(new_paragraphs)
            base_64 = ima.base64_from_icon(icons[sev], size, size)
            if max_lines_msglist >= 0:
                msglist.append(template.format(base_64, msg, src, code, size=size))
        if msglist:
            self.show_tooltip(title=_('Code analysis'), text='\n'.join(msglist), title_color=QStylePalette.COLOR_ACCENT_4, at_line=line_number, with_html_format=True)
            self.highlight_line_warning(block_data)

    def highlight_line_warning(self, block_data):
        if False:
            i = 10
            return i + 15
        'Highlight errors and warnings in this editor.'
        self.clear_extra_selections('code_analysis_highlight')
        self.highlight_selection('code_analysis_highlight', block_data._selection(), background_color=block_data.color)
        self.linenumberarea.update()

    def get_current_warnings(self):
        if False:
            while True:
                i = 10
        '\n        Get all warnings for the current editor and return\n        a list with the message and line number.\n        '
        block = self.document().firstBlock()
        line_count = self.document().blockCount()
        warnings = []
        while True:
            data = block.userData()
            if data and data.code_analysis:
                for warning in data.code_analysis:
                    warnings.append([warning[-1], block.blockNumber() + 1])
            if block.blockNumber() + 1 == line_count:
                break
            block = block.next()
        return warnings

    def go_to_next_warning(self):
        if False:
            print('Hello World!')
        '\n        Go to next code warning message and return new cursor position.\n        '
        block = self.textCursor().block()
        line_count = self.document().blockCount()
        for __ in range(line_count):
            line_number = block.blockNumber() + 1
            if line_number < line_count:
                block = block.next()
            else:
                block = self.document().firstBlock()
            data = block.userData()
            if data and data.code_analysis:
                line_number = block.blockNumber() + 1
                self.go_to_line(line_number)
                self.show_code_analysis_results(line_number, data)
                return self.get_position('cursor')

    def go_to_previous_warning(self):
        if False:
            while True:
                i = 10
        '\n        Go to previous code warning message and return new cursor position.\n        '
        block = self.textCursor().block()
        line_count = self.document().blockCount()
        for __ in range(line_count):
            line_number = block.blockNumber() + 1
            if line_number > 1:
                block = block.previous()
            else:
                block = self.document().lastBlock()
            data = block.userData()
            if data and data.code_analysis:
                line_number = block.blockNumber() + 1
                self.go_to_line(line_number)
                self.show_code_analysis_results(line_number, data)
                return self.get_position('cursor')

    def cell_list(self):
        if False:
            print('Hello World!')
        'Get the outline explorer data for all cells.'
        for oedata in self.outlineexplorer_data_list():
            if oedata.def_type == OED.CELL:
                yield oedata

    def get_cell_code(self, cell):
        if False:
            while True:
                i = 10
        "\n        Get cell code for a given cell.\n\n        If the cell doesn't exist, raises an exception\n        "
        selected_block = None
        if is_string(cell):
            for oedata in self.cell_list():
                if oedata.def_name == cell:
                    selected_block = oedata.block
                    break
        elif cell == 0:
            selected_block = self.document().firstBlock()
        else:
            cell_list = list(self.cell_list())
            if cell <= len(cell_list):
                selected_block = cell_list[cell - 1].block
        if not selected_block:
            raise RuntimeError('Cell {} not found.'.format(repr(cell)))
        cursor = QTextCursor(selected_block)
        (text, _, off_pos, col_pos) = self.get_cell_as_executable_code(cursor)
        return text

    def get_cell_code_and_position(self, cell):
        if False:
            i = 10
            return i + 15
        "\n        Get code and position for a given cell.\n\n        If the cell doesn't exist, raise an exception.\n        "
        selected_block = None
        if is_string(cell):
            for oedata in self.cell_list():
                if oedata.def_name == cell:
                    selected_block = oedata.block
                    break
        elif cell == 0:
            selected_block = self.document().firstBlock()
        else:
            cell_list = list(self.cell_list())
            if cell <= len(cell_list):
                selected_block = cell_list[cell - 1].block
        if not selected_block:
            raise RuntimeError('Cell {} not found.'.format(repr(cell)))
        cursor = QTextCursor(selected_block)
        (text, _, off_pos, col_pos) = self.get_cell_as_executable_code(cursor)
        return (text, off_pos, col_pos)

    def get_cell_count(self):
        if False:
            while True:
                i = 10
        'Get number of cells in document.'
        return 1 + len(list(self.cell_list()))

    def go_to_next_todo(self):
        if False:
            for i in range(10):
                print('nop')
        'Go to next todo and return new cursor position'
        block = self.textCursor().block()
        line_count = self.document().blockCount()
        while True:
            if block.blockNumber() + 1 < line_count:
                block = block.next()
            else:
                block = self.document().firstBlock()
            data = block.userData()
            if data and data.todo:
                break
        line_number = block.blockNumber() + 1
        self.go_to_line(line_number)
        self.show_tooltip(title=_('To do'), text=data.todo, title_color=QStylePalette.COLOR_ACCENT_4, at_line=line_number)
        return self.get_position('cursor')

    def process_todo(self, todo_results):
        if False:
            return 10
        'Process todo finder results'
        for data in self.blockuserdata_list():
            data.todo = ''
        for (message, line_number) in todo_results:
            block = self.document().findBlockByNumber(line_number - 1)
            data = block.userData()
            if not data:
                data = BlockUserData(self)
            data.todo = message
            block.setUserData(data)
        self.sig_flags_changed.emit()

    def add_prefix(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        'Add prefix to current line or selected line(s)'
        cursor = self.textCursor()
        if self.has_selected_text():
            (start_pos, end_pos) = (cursor.selectionStart(), cursor.selectionEnd())
            first_pos = min([start_pos, end_pos])
            first_cursor = self.textCursor()
            first_cursor.setPosition(first_pos)
            cursor.beginEditBlock()
            cursor.setPosition(end_pos)
            if cursor.atBlockStart():
                cursor.movePosition(QTextCursor.PreviousBlock)
                if cursor.position() < start_pos:
                    cursor.setPosition(start_pos)
            move_number = self.__spaces_for_prefix()
            while cursor.position() >= start_pos:
                cursor.movePosition(QTextCursor.StartOfBlock)
                line_text = to_text_string(cursor.block().text())
                if self.get_character(cursor.position()) == ' ' and '#' in prefix and (not line_text.isspace()) or (not line_text.startswith(' ') and line_text != ''):
                    cursor.movePosition(QTextCursor.Right, QTextCursor.MoveAnchor, move_number)
                    cursor.insertText(prefix)
                elif '#' not in prefix:
                    cursor.insertText(prefix)
                if cursor.blockNumber() == 0:
                    break
                cursor.movePosition(QTextCursor.PreviousBlock)
                cursor.movePosition(QTextCursor.EndOfBlock)
            cursor.endEditBlock()
        else:
            cursor.beginEditBlock()
            cursor.movePosition(QTextCursor.StartOfBlock)
            if self.get_character(cursor.position()) == ' ' and '#' in prefix:
                cursor.movePosition(QTextCursor.NextWord)
            cursor.insertText(prefix)
            cursor.endEditBlock()

    def __spaces_for_prefix(self):
        if False:
            return 10
        'Find the less indented level of text.'
        cursor = self.textCursor()
        if self.has_selected_text():
            (start_pos, end_pos) = (cursor.selectionStart(), cursor.selectionEnd())
            first_pos = min([start_pos, end_pos])
            first_cursor = self.textCursor()
            first_cursor.setPosition(first_pos)
            cursor.beginEditBlock()
            cursor.setPosition(end_pos)
            if cursor.atBlockStart():
                cursor.movePosition(QTextCursor.PreviousBlock)
                if cursor.position() < start_pos:
                    cursor.setPosition(start_pos)
            number_spaces = -1
            while cursor.position() >= start_pos:
                cursor.movePosition(QTextCursor.StartOfBlock)
                line_text = to_text_string(cursor.block().text())
                start_with_space = line_text.startswith(' ')
                left_number_spaces = self.__number_of_spaces(line_text)
                if not start_with_space:
                    left_number_spaces = 0
                if (number_spaces == -1 or number_spaces > left_number_spaces) and (not line_text.isspace()) and (line_text != ''):
                    number_spaces = left_number_spaces
                if cursor.blockNumber() == 0:
                    break
                cursor.movePosition(QTextCursor.PreviousBlock)
                cursor.movePosition(QTextCursor.EndOfBlock)
            cursor.endEditBlock()
        return number_spaces

    def remove_suffix(self, suffix):
        if False:
            i = 10
            return i + 15
        '\n        Remove suffix from current line (there should not be any selection)\n        '
        cursor = self.textCursor()
        cursor.setPosition(cursor.position() - qstring_length(suffix), QTextCursor.KeepAnchor)
        if to_text_string(cursor.selectedText()) == suffix:
            cursor.removeSelectedText()

    def remove_prefix(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        'Remove prefix from current line or selected line(s)'
        cursor = self.textCursor()
        if self.has_selected_text():
            (start_pos, end_pos) = sorted([cursor.selectionStart(), cursor.selectionEnd()])
            cursor.setPosition(start_pos)
            if not cursor.atBlockStart():
                cursor.movePosition(QTextCursor.StartOfBlock)
                start_pos = cursor.position()
            cursor.beginEditBlock()
            cursor.setPosition(end_pos)
            if cursor.atBlockStart():
                cursor.movePosition(QTextCursor.PreviousBlock)
                if cursor.position() < start_pos:
                    cursor.setPosition(start_pos)
            cursor.movePosition(QTextCursor.StartOfBlock)
            old_pos = None
            while cursor.position() >= start_pos:
                new_pos = cursor.position()
                if old_pos == new_pos:
                    break
                else:
                    old_pos = new_pos
                line_text = to_text_string(cursor.block().text())
                self.__remove_prefix(prefix, cursor, line_text)
                cursor.movePosition(QTextCursor.PreviousBlock)
            cursor.endEditBlock()
        else:
            cursor.movePosition(QTextCursor.StartOfBlock)
            line_text = to_text_string(cursor.block().text())
            self.__remove_prefix(prefix, cursor, line_text)

    def __remove_prefix(self, prefix, cursor, line_text):
        if False:
            for i in range(10):
                print('nop')
        'Handle the removal of the prefix for a single line.'
        start_with_space = line_text.startswith(' ')
        if start_with_space:
            left_spaces = self.__even_number_of_spaces(line_text)
        else:
            left_spaces = False
        if start_with_space:
            right_number_spaces = self.__number_of_spaces(line_text, group=1)
        else:
            right_number_spaces = self.__number_of_spaces(line_text)
        if prefix.strip() and line_text.lstrip().startswith(prefix + ' ') or (line_text.startswith(prefix + ' ') and '#' in prefix):
            cursor.movePosition(QTextCursor.Right, QTextCursor.MoveAnchor, line_text.find(prefix))
            if right_number_spaces == 1 and (left_spaces or not start_with_space) or (not start_with_space and right_number_spaces % 2 != 0) or (left_spaces and right_number_spaces % 2 != 0):
                cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, len(prefix + ' '))
            else:
                cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, len(prefix))
            cursor.removeSelectedText()
        elif prefix.strip() and line_text.lstrip().startswith(prefix) or line_text.startswith(prefix):
            cursor.movePosition(QTextCursor.Right, QTextCursor.MoveAnchor, line_text.find(prefix))
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, len(prefix))
            cursor.removeSelectedText()

    def __even_number_of_spaces(self, line_text, group=0):
        if False:
            while True:
                i = 10
        '\n        Get if there is a correct indentation from a group of spaces of a line.\n        '
        spaces = re.findall('\\s+', line_text)
        if len(spaces) - 1 >= group:
            return len(spaces[group]) % len(self.indent_chars) == 0

    def __number_of_spaces(self, line_text, group=0):
        if False:
            print('Hello World!')
        'Get the number of spaces from a group of spaces in a line.'
        spaces = re.findall('\\s+', line_text)
        if len(spaces) - 1 >= group:
            return len(spaces[group])

    def __get_brackets(self, line_text, closing_brackets=[]):
        if False:
            while True:
                i = 10
        "\n        Return unmatched opening brackets and left-over closing brackets.\n\n        (str, []) -> ([(pos, bracket)], [bracket], comment_pos)\n\n        Iterate through line_text to find unmatched brackets.\n\n        Returns three objects as a tuple:\n        1) bracket_stack:\n            a list of tuples of pos and char of each unmatched opening bracket\n        2) closing brackets:\n            this line's unmatched closing brackets + arg closing_brackets.\n            If this line ad no closing brackets, arg closing_brackets might\n            be matched with previously unmatched opening brackets in this line.\n        3) Pos at which a # comment begins. -1 if it doesn't.'\n        "
        bracket_stack = []
        bracket_unmatched_closing = []
        comment_pos = -1
        deactivate = None
        escaped = False
        (pos, c) = (None, None)
        for (pos, c) in enumerate(line_text):
            if escaped:
                escaped = False
            elif deactivate:
                if c == deactivate:
                    deactivate = None
                elif c == '\\':
                    escaped = True
            elif c in ["'", '"']:
                deactivate = c
            elif c == '#':
                comment_pos = pos
                break
            elif c in ('(', '[', '{'):
                bracket_stack.append((pos, c))
            elif c in (')', ']', '}'):
                if bracket_stack and bracket_stack[-1][1] == {')': '(', ']': '[', '}': '{'}[c]:
                    bracket_stack.pop()
                else:
                    bracket_unmatched_closing.append(c)
        del pos, deactivate, escaped
        if not bracket_unmatched_closing:
            for c in list(closing_brackets):
                if bracket_stack and bracket_stack[-1][1] == {')': '(', ']': '[', '}': '{'}[c]:
                    bracket_stack.pop()
                    closing_brackets.remove(c)
                else:
                    break
        del c
        closing_brackets = bracket_unmatched_closing + closing_brackets
        return (bracket_stack, closing_brackets, comment_pos)

    def fix_indent(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Indent line according to the preferences'
        if self.is_python_like():
            ret = self.fix_indent_smart(*args, **kwargs)
        else:
            ret = self.simple_indentation(*args, **kwargs)
        return ret

    def simple_indentation(self, forward=True, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Simply preserve the indentation-level of the previous line.\n        '
        cursor = self.textCursor()
        block_nb = cursor.blockNumber()
        prev_block = self.document().findBlockByNumber(block_nb - 1)
        prevline = to_text_string(prev_block.text())
        indentation = re.match('\\s*', prevline).group()
        if not forward:
            indentation = indentation[len(self.indent_chars):]
        cursor.insertText(indentation)
        return False

    def find_indentation(self, forward=True, comment_or_string=False, cur_indent=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find indentation (Python only, no text selection)\n\n        forward=True: fix indent only if text is not enough indented\n                      (otherwise force indent)\n        forward=False: fix indent only if text is too much indented\n                       (otherwise force unindent)\n\n        comment_or_string: Do not adjust indent level for\n            unmatched opening brackets and keywords\n\n        max_blank_lines: maximum number of blank lines to search before giving\n            up\n\n        cur_indent: current indent. This is the indent before we started\n            processing. E.g. when returning, indent before rstrip.\n\n        Returns the indentation for the current line\n\n        Assumes self.is_python_like() to return True\n        '
        cursor = self.textCursor()
        block_nb = cursor.blockNumber()
        line_in_block = False
        visual_indent = False
        add_indent = 0
        prevline = None
        prevtext = ''
        empty_lines = True
        closing_brackets = []
        for prevline in range(block_nb - 1, -1, -1):
            cursor.movePosition(QTextCursor.PreviousBlock)
            prevtext = to_text_string(cursor.block().text()).rstrip()
            (bracket_stack, closing_brackets, comment_pos) = self.__get_brackets(prevtext, closing_brackets)
            if not prevtext:
                continue
            if prevtext.endswith((':', '\\')):
                line_in_block = True
            if bracket_stack or not closing_brackets:
                break
            if prevtext.strip() != '':
                empty_lines = False
        if empty_lines and prevline is not None and (prevline < block_nb - 2):
            prevtext = ''
            prevline = block_nb - 2
            line_in_block = False
        words = re.split('[\\s\\(\\[\\{\\}\\]\\)]', prevtext.lstrip())
        if line_in_block:
            add_indent += 1
        if prevtext and (not comment_or_string):
            if bracket_stack:
                if prevtext.endswith(('(', '[', '{')):
                    add_indent += 1
                    if words[0] in ('class', 'def', 'elif', 'except', 'for', 'if', 'while', 'with'):
                        add_indent += 1
                    elif not (self.tab_stop_width_spaces if self.indent_chars == '\t' else len(self.indent_chars)) * 2 < len(prevtext):
                        visual_indent = True
                else:
                    visual_indent = True
            elif words[-1] in ('continue', 'break', 'pass') or (words[0] == 'return' and (not line_in_block)):
                add_indent -= 1
        if prevline:
            prevline_indent = self.get_block_indentation(prevline)
        else:
            prevline_indent = 0
        if visual_indent:
            correct_indent = bracket_stack[-1][0] + 1
        elif add_indent:
            if self.indent_chars == '\t':
                correct_indent = prevline_indent + self.tab_stop_width_spaces * add_indent
            else:
                correct_indent = prevline_indent + len(self.indent_chars) * add_indent
        else:
            correct_indent = prevline_indent
        if prevline and (not bracket_stack) and (not prevtext.endswith(':')):
            if forward:
                ref_line = block_nb - 1
            else:
                ref_line = prevline
            if cur_indent is None:
                cur_indent = self.get_block_indentation(ref_line)
            is_blank = not self.get_text_line(ref_line).strip()
            trailing_text = self.get_text_line(block_nb).strip()
            if cur_indent < prevline_indent and (trailing_text or is_blank):
                correct_indent = -(-cur_indent // len(self.indent_chars)) * len(self.indent_chars)
        return correct_indent

    def fix_indent_smart(self, forward=True, comment_or_string=False, cur_indent=None):
        if False:
            return 10
        '\n        Fix indentation (Python only, no text selection)\n\n        forward=True: fix indent only if text is not enough indented\n                      (otherwise force indent)\n        forward=False: fix indent only if text is too much indented\n                       (otherwise force unindent)\n\n        comment_or_string: Do not adjust indent level for\n            unmatched opening brackets and keywords\n\n        max_blank_lines: maximum number of blank lines to search before giving\n            up\n\n        cur_indent: current indent. This is the indent before we started\n            processing. E.g. when returning, indent before rstrip.\n\n        Returns True if indent needed to be fixed\n\n        Assumes self.is_python_like() to return True\n        '
        cursor = self.textCursor()
        block_nb = cursor.blockNumber()
        indent = self.get_block_indentation(block_nb)
        correct_indent = self.find_indentation(forward, comment_or_string, cur_indent)
        if correct_indent >= 0 and (not (indent == correct_indent or (forward and indent > correct_indent) or (not forward and indent < correct_indent))):
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.StartOfBlock)
            if self.indent_chars == '\t':
                indent = indent // self.tab_stop_width_spaces
            cursor.setPosition(cursor.position() + indent, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            if self.indent_chars == '\t':
                indent_text = '\t' * (correct_indent // self.tab_stop_width_spaces) + ' ' * (correct_indent % self.tab_stop_width_spaces)
            else:
                indent_text = ' ' * correct_indent
            cursor.insertText(indent_text)
            return True
        return False

    @Slot()
    def clear_all_output(self):
        if False:
            i = 10
            return i + 15
        'Removes all output in the ipynb format (Json only)'
        try:
            nb = nbformat.reads(self.toPlainText(), as_version=4)
            if nb.cells:
                for cell in nb.cells:
                    if 'outputs' in cell:
                        cell['outputs'] = []
                    if 'prompt_number' in cell:
                        cell['prompt_number'] = None
            self.selectAll()
            self.skip_rstrip = True
            self.insertPlainText(nbformat.writes(nb))
            self.skip_rstrip = False
        except Exception as e:
            QMessageBox.critical(self, _('Removal error'), _('It was not possible to remove outputs from this notebook. The error is:\n\n') + to_text_string(e))
            return

    @Slot()
    def convert_notebook(self):
        if False:
            print('Hello World!')
        'Convert an IPython notebook to a Python script in editor'
        try:
            nb = nbformat.reads(self.toPlainText(), as_version=4)
            script = nbexporter().from_notebook_node(nb)[0]
        except Exception as e:
            QMessageBox.critical(self, _('Conversion error'), _('It was not possible to convert this notebook. The error is:\n\n') + to_text_string(e))
            return
        self.sig_new_file.emit(script)

    def indent(self, force=False):
        if False:
            while True:
                i = 10
        '\n        Indent current line or selection\n\n        force=True: indent even if cursor is not a the beginning of the line\n        '
        leading_text = self.get_text('sol', 'cursor')
        if self.has_selected_text():
            self.add_prefix(self.indent_chars)
        elif force or not leading_text.strip() or (self.tab_indents and self.tab_mode):
            if self.is_python_like():
                if not self.fix_indent(forward=True):
                    self.add_prefix(self.indent_chars)
            else:
                self.add_prefix(self.indent_chars)
        elif len(self.indent_chars) > 1:
            length = len(self.indent_chars)
            self.insert_text(' ' * (length - len(leading_text) % length))
        else:
            self.insert_text(self.indent_chars)

    def indent_or_replace(self):
        if False:
            i = 10
            return i + 15
        'Indent or replace by 4 spaces depending on selection and tab mode'
        if self.tab_indents and self.tab_mode or not self.has_selected_text():
            self.indent()
        else:
            cursor = self.textCursor()
            if self.get_selected_text() == to_text_string(cursor.block().text()):
                self.indent()
            else:
                cursor1 = self.textCursor()
                cursor1.setPosition(cursor.selectionStart())
                cursor2 = self.textCursor()
                cursor2.setPosition(cursor.selectionEnd())
                if cursor1.blockNumber() != cursor2.blockNumber():
                    self.indent()
                else:
                    self.replace(self.indent_chars)

    def unindent(self, force=False):
        if False:
            return 10
        '\n        Unindent current line or selection\n\n        force=True: unindent even if cursor is not a the beginning of the line\n        '
        if self.has_selected_text():
            if self.indent_chars == '\t':
                self.remove_prefix(self.indent_chars)
            else:
                space_count = len(self.indent_chars)
                leading_spaces = self.__spaces_for_prefix()
                remainder = leading_spaces % space_count
                if remainder:
                    self.remove_prefix(' ' * remainder)
                else:
                    self.remove_prefix(self.indent_chars)
        else:
            leading_text = self.get_text('sol', 'cursor')
            if force or not leading_text.strip() or (self.tab_indents and self.tab_mode):
                if self.is_python_like():
                    if not self.fix_indent(forward=False):
                        self.remove_prefix(self.indent_chars)
                elif leading_text.endswith('\t'):
                    self.remove_prefix('\t')
                else:
                    self.remove_prefix(self.indent_chars)

    @Slot()
    def toggle_comment(self):
        if False:
            i = 10
            return i + 15
        'Toggle comment on current line or selection'
        cursor = self.textCursor()
        (start_pos, end_pos) = sorted([cursor.selectionStart(), cursor.selectionEnd()])
        cursor.setPosition(end_pos)
        last_line = cursor.block().blockNumber()
        if cursor.atBlockStart() and start_pos != end_pos:
            last_line -= 1
        cursor.setPosition(start_pos)
        first_line = cursor.block().blockNumber()
        is_comment_or_whitespace = True
        at_least_one_comment = False
        for _line_nb in range(first_line, last_line + 1):
            text = to_text_string(cursor.block().text()).lstrip()
            is_comment = text.startswith(self.comment_string)
            is_whitespace = text == ''
            is_comment_or_whitespace *= is_comment or is_whitespace
            if is_comment:
                at_least_one_comment = True
            cursor.movePosition(QTextCursor.NextBlock)
        if is_comment_or_whitespace and at_least_one_comment:
            self.uncomment()
        else:
            self.comment()

    def is_comment(self, block):
        if False:
            print('Hello World!')
        'Detect inline comments.\n\n        Return True if the block is an inline comment.\n        '
        if block is None:
            return False
        text = to_text_string(block.text()).lstrip()
        return text.startswith(self.comment_string)

    def comment(self):
        if False:
            for i in range(10):
                print('nop')
        'Comment current line or selection.'
        self.add_prefix(self.comment_string + ' ')

    def uncomment(self):
        if False:
            for i in range(10):
                print('nop')
        'Uncomment current line or selection.'
        blockcomment = self.unblockcomment()
        if not blockcomment:
            self.remove_prefix(self.comment_string)

    def __blockcomment_bar(self, compatibility=False):
        if False:
            return 10
        'Handle versions of blockcomment bar for backwards compatibility.'
        blockcomment_bar = self.comment_string + ' ' + '=' * (79 - len(self.comment_string + ' '))
        if compatibility:
            blockcomment_bar = self.comment_string + '=' * (79 - len(self.comment_string))
        return blockcomment_bar

    def transform_to_uppercase(self):
        if False:
            return 10
        'Change to uppercase current line or selection.'
        cursor = self.textCursor()
        prev_pos = cursor.position()
        selected_text = to_text_string(cursor.selectedText())
        if len(selected_text) == 0:
            prev_pos = cursor.position()
            cursor.select(QTextCursor.WordUnderCursor)
            selected_text = to_text_string(cursor.selectedText())
        s = selected_text.upper()
        cursor.insertText(s)
        self.set_cursor_position(prev_pos)

    def transform_to_lowercase(self):
        if False:
            for i in range(10):
                print('nop')
        'Change to lowercase current line or selection.'
        cursor = self.textCursor()
        prev_pos = cursor.position()
        selected_text = to_text_string(cursor.selectedText())
        if len(selected_text) == 0:
            prev_pos = cursor.position()
            cursor.select(QTextCursor.WordUnderCursor)
            selected_text = to_text_string(cursor.selectedText())
        s = selected_text.lower()
        cursor.insertText(s)
        self.set_cursor_position(prev_pos)

    def blockcomment(self):
        if False:
            print('Hello World!')
        'Block comment current line or selection.'
        comline = self.__blockcomment_bar() + self.get_line_separator()
        cursor = self.textCursor()
        if self.has_selected_text():
            self.extend_selection_to_complete_lines()
            (start_pos, end_pos) = (cursor.selectionStart(), cursor.selectionEnd())
        else:
            start_pos = end_pos = cursor.position()
        cursor.beginEditBlock()
        cursor.setPosition(start_pos)
        cursor.movePosition(QTextCursor.StartOfBlock)
        while cursor.position() <= end_pos:
            cursor.insertText(self.comment_string + ' ')
            cursor.movePosition(QTextCursor.EndOfBlock)
            if cursor.atEnd():
                break
            cursor.movePosition(QTextCursor.NextBlock)
            end_pos += len(self.comment_string + ' ')
        cursor.setPosition(end_pos)
        cursor.movePosition(QTextCursor.EndOfBlock)
        if cursor.atEnd():
            cursor.insertText(self.get_line_separator())
        else:
            cursor.movePosition(QTextCursor.NextBlock)
        cursor.insertText(comline)
        cursor.setPosition(start_pos)
        cursor.movePosition(QTextCursor.StartOfBlock)
        cursor.insertText(comline)
        cursor.endEditBlock()

    def unblockcomment(self):
        if False:
            for i in range(10):
                print('nop')
        'Un-block comment current line or selection.'
        unblockcomment = self.__unblockcomment()
        if not unblockcomment:
            unblockcomment = self.__unblockcomment(compatibility=True)
        else:
            return unblockcomment

    def __unblockcomment(self, compatibility=False):
        if False:
            while True:
                i = 10
        'Un-block comment current line or selection helper.'

        def __is_comment_bar(cursor):
            if False:
                print('Hello World!')
            return to_text_string(cursor.block().text()).startswith(self.__blockcomment_bar(compatibility=compatibility))
        cursor1 = self.textCursor()
        if __is_comment_bar(cursor1):
            return
        while not __is_comment_bar(cursor1):
            cursor1.movePosition(QTextCursor.PreviousBlock)
            if cursor1.blockNumber() == 0:
                break
        if not __is_comment_bar(cursor1):
            return False

        def __in_block_comment(cursor):
            if False:
                for i in range(10):
                    print('nop')
            cs = self.comment_string
            return to_text_string(cursor.block().text()).startswith(cs)
        cursor2 = QTextCursor(cursor1)
        cursor2.movePosition(QTextCursor.NextBlock)
        while not __is_comment_bar(cursor2) and __in_block_comment(cursor2):
            cursor2.movePosition(QTextCursor.NextBlock)
            if cursor2.block() == self.document().lastBlock():
                break
        if not __is_comment_bar(cursor2):
            return False
        cursor3 = self.textCursor()
        cursor3.beginEditBlock()
        cursor3.setPosition(cursor1.position())
        cursor3.movePosition(QTextCursor.NextBlock)
        while cursor3.position() < cursor2.position():
            cursor3.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
            if not cursor3.atBlockEnd():
                if not compatibility:
                    cursor3.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
            cursor3.removeSelectedText()
            cursor3.movePosition(QTextCursor.NextBlock)
        for cursor in (cursor2, cursor1):
            cursor3.setPosition(cursor.position())
            cursor3.select(QTextCursor.BlockUnderCursor)
            cursor3.removeSelectedText()
        cursor3.endEditBlock()
        return True

    def create_new_cell(self):
        if False:
            return 10
        firstline = '# %%' + self.get_line_separator()
        endline = self.get_line_separator()
        cursor = self.textCursor()
        if self.has_selected_text():
            self.extend_selection_to_complete_lines()
            (start_pos, end_pos) = (cursor.selectionStart(), cursor.selectionEnd())
            endline = self.get_line_separator() + '# %%'
        else:
            start_pos = end_pos = cursor.position()
        cursor.beginEditBlock()
        cursor.setPosition(end_pos)
        cursor.movePosition(QTextCursor.EndOfBlock)
        cursor.insertText(endline)
        cursor.setPosition(start_pos)
        cursor.movePosition(QTextCursor.StartOfBlock)
        cursor.insertText(firstline)
        cursor.endEditBlock()

    def kill_line_end(self):
        if False:
            while True:
                i = 10
        'Kill the text on the current line from the cursor forward'
        cursor = self.textCursor()
        cursor.clearSelection()
        cursor.movePosition(QTextCursor.EndOfLine, QTextCursor.KeepAnchor)
        if not cursor.hasSelection():
            cursor.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor)
        self._kill_ring.kill_cursor(cursor)
        self.setTextCursor(cursor)

    def kill_line_start(self):
        if False:
            for i in range(10):
                print('nop')
        'Kill the text on the current line from the cursor backward'
        cursor = self.textCursor()
        cursor.clearSelection()
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        self._kill_ring.kill_cursor(cursor)
        self.setTextCursor(cursor)

    def _get_word_start_cursor(self, position):
        if False:
            return 10
        'Find the start of the word to the left of the given position. If a\n           sequence of non-word characters precedes the first word, skip over\n           them. (This emulates the behavior of bash, emacs, etc.)\n        '
        document = self.document()
        position -= 1
        while position and (not self.is_letter_or_number(document.characterAt(position))):
            position -= 1
        while position and self.is_letter_or_number(document.characterAt(position)):
            position -= 1
        cursor = self.textCursor()
        cursor.setPosition(self.next_cursor_position())
        return cursor

    def _get_word_end_cursor(self, position):
        if False:
            for i in range(10):
                print('nop')
        'Find the end of the word to the right of the given position. If a\n           sequence of non-word characters precedes the first word, skip over\n           them. (This emulates the behavior of bash, emacs, etc.)\n        '
        document = self.document()
        cursor = self.textCursor()
        position = cursor.position()
        cursor.movePosition(QTextCursor.End)
        end = cursor.position()
        while position < end and (not self.is_letter_or_number(document.characterAt(position))):
            position = self.next_cursor_position(position)
        while position < end and self.is_letter_or_number(document.characterAt(position)):
            position = self.next_cursor_position(position)
        cursor.setPosition(position)
        return cursor

    def kill_prev_word(self):
        if False:
            return 10
        'Kill the previous word'
        position = self.textCursor().position()
        cursor = self._get_word_start_cursor(position)
        cursor.setPosition(position, QTextCursor.KeepAnchor)
        self._kill_ring.kill_cursor(cursor)
        self.setTextCursor(cursor)

    def kill_next_word(self):
        if False:
            for i in range(10):
                print('nop')
        'Kill the next word'
        position = self.textCursor().position()
        cursor = self._get_word_end_cursor(position)
        cursor.setPosition(position, QTextCursor.KeepAnchor)
        self._kill_ring.kill_cursor(cursor)
        self.setTextCursor(cursor)

    def __get_current_color(self, cursor=None):
        if False:
            i = 10
            return i + 15
        'Get the syntax highlighting color for the current cursor position'
        if cursor is None:
            cursor = self.textCursor()
        block = cursor.block()
        pos = cursor.position() - block.position()
        layout = block.layout()
        block_formats = layout.additionalFormats()
        if block_formats:
            if cursor.atBlockEnd():
                current_format = block_formats[-1].format
            else:
                current_format = None
                for fmt in block_formats:
                    if pos >= fmt.start and pos < fmt.start + fmt.length:
                        current_format = fmt.format
                if current_format is None:
                    return None
            color = current_format.foreground().color().name()
            return color
        else:
            return None

    def in_comment_or_string(self, cursor=None, position=None):
        if False:
            for i in range(10):
                print('nop')
        'Is the cursor or position inside or next to a comment or string?\n\n        If *cursor* is None, *position* is used instead. If *position* is also\n        None, then the current cursor position is used.\n        '
        if self.highlighter:
            if cursor is None:
                cursor = self.textCursor()
                if position:
                    cursor.setPosition(position)
            current_color = self.__get_current_color(cursor=cursor)
            comment_color = self.highlighter.get_color_name('comment')
            string_color = self.highlighter.get_color_name('string')
            if current_color == comment_color or current_color == string_color:
                return True
            else:
                return False
        else:
            return False

    def __colon_keyword(self, text):
        if False:
            return 10
        stmt_kws = ['def', 'for', 'if', 'while', 'with', 'class', 'elif', 'except']
        whole_kws = ['else', 'try', 'except', 'finally']
        text = text.lstrip()
        words = text.split()
        if any([text == wk for wk in whole_kws]):
            return True
        elif len(words) < 2:
            return False
        elif any([words[0] == sk for sk in stmt_kws]):
            return True
        else:
            return False

    def __forbidden_colon_end_char(self, text):
        if False:
            i = 10
            return i + 15
        end_chars = [':', '\\', '[', '{', '(', ',']
        text = text.rstrip()
        if any([text.endswith(c) for c in end_chars]):
            return True
        else:
            return False

    def __has_colon_not_in_brackets(self, text):
        if False:
            return 10
        '\n        Return whether a string has a colon which is not between brackets.\n        This function returns True if the given string has a colon which is\n        not between a pair of (round, square or curly) brackets. It assumes\n        that the brackets in the string are balanced.\n        '
        bracket_ext = self.editor_extensions.get(CloseBracketsExtension)
        for (pos, char) in enumerate(text):
            if char == ':' and (not bracket_ext.unmatched_brackets_in_line(text[:pos])):
                return True
        return False

    def __has_unmatched_opening_bracket(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if there are any unmatched opening brackets before the current\n        cursor position.\n        '
        position = self.textCursor().position()
        for brace in [']', ')', '}']:
            match = self.find_brace_match(position, brace, forward=False)
            if match is not None:
                return True
        return False

    def autoinsert_colons(self):
        if False:
            return 10
        'Decide if we want to autoinsert colons'
        bracket_ext = self.editor_extensions.get(CloseBracketsExtension)
        self.completion_widget.hide()
        line_text = self.get_text('sol', 'cursor')
        if not self.textCursor().atBlockEnd():
            return False
        elif self.in_comment_or_string():
            return False
        elif not self.__colon_keyword(line_text):
            return False
        elif self.__forbidden_colon_end_char(line_text):
            return False
        elif bracket_ext.unmatched_brackets_in_line(line_text):
            return False
        elif self.__has_colon_not_in_brackets(line_text):
            return False
        elif self.__has_unmatched_opening_bracket():
            return False
        else:
            return True

    def next_char(self):
        if False:
            i = 10
            return i + 15
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
        next_char = to_text_string(cursor.selectedText())
        return next_char

    def in_comment(self, cursor=None, position=None):
        if False:
            return 10
        'Returns True if the given position is inside a comment.\n\n        Parameters\n        ----------\n        cursor : QTextCursor, optional\n            The position to check.\n        position : int, optional\n            The position to check if *cursor* is None. This parameter\n            is ignored when *cursor* is not None.\n\n        If both *cursor* and *position* are none, then the position returned\n        by self.textCursor() is used instead.\n        '
        if self.highlighter:
            if cursor is None:
                cursor = self.textCursor()
                if position is not None:
                    cursor.setPosition(position)
            current_color = self.__get_current_color(cursor)
            comment_color = self.highlighter.get_color_name('comment')
            return current_color == comment_color
        else:
            return False

    def in_string(self, cursor=None, position=None):
        if False:
            i = 10
            return i + 15
        'Returns True if the given position is inside a string.\n\n        Parameters\n        ----------\n        cursor : QTextCursor, optional\n            The position to check.\n        position : int, optional\n            The position to check if *cursor* is None. This parameter\n            is ignored when *cursor* is not None.\n\n        If both *cursor* and *position* are none, then the position returned\n        by self.textCursor() is used instead.\n        '
        if self.highlighter:
            if cursor is None:
                cursor = self.textCursor()
                if position is not None:
                    cursor.setPosition(position)
            current_color = self.__get_current_color(cursor)
            string_color = self.highlighter.get_color_name('string')
            return current_color == string_color
        else:
            return False

    def setup_context_menu(self):
        if False:
            print('Hello World!')
        'Setup context menu'
        self.undo_action = create_action(self, _('Undo'), icon=ima.icon('undo'), shortcut=self.get_shortcut('undo'), triggered=self.undo)
        self.redo_action = create_action(self, _('Redo'), icon=ima.icon('redo'), shortcut=self.get_shortcut('redo'), triggered=self.redo)
        self.cut_action = create_action(self, _('Cut'), icon=ima.icon('editcut'), shortcut=self.get_shortcut('cut'), triggered=self.cut)
        self.copy_action = create_action(self, _('Copy'), icon=ima.icon('editcopy'), shortcut=self.get_shortcut('copy'), triggered=self.copy)
        self.paste_action = create_action(self, _('Paste'), icon=ima.icon('editpaste'), shortcut=self.get_shortcut('paste'), triggered=self.paste)
        selectall_action = create_action(self, _('Select All'), icon=ima.icon('selectall'), shortcut=self.get_shortcut('select all'), triggered=self.selectAll)
        toggle_comment_action = create_action(self, _('Comment') + '/' + _('Uncomment'), icon=ima.icon('comment'), shortcut=self.get_shortcut('toggle comment'), triggered=self.toggle_comment)
        self.clear_all_output_action = create_action(self, _('Clear all ouput'), icon=ima.icon('ipython_console'), triggered=self.clear_all_output)
        self.ipynb_convert_action = create_action(self, _('Convert to Python file'), icon=ima.icon('python'), triggered=self.convert_notebook)
        self.gotodef_action = create_action(self, _('Go to definition'), shortcut=self.get_shortcut('go to definition'), triggered=self.go_to_definition_from_cursor)
        self.inspect_current_object_action = create_action(self, _('Inspect current object'), icon=ima.icon('MessageBoxInformation'), shortcut=self.get_shortcut('inspect current object'), triggered=self.sig_show_object_info)
        zoom_in_action = create_action(self, _('Zoom in'), icon=ima.icon('zoom_in'), shortcut=QKeySequence(QKeySequence.ZoomIn), triggered=self.zoom_in)
        zoom_out_action = create_action(self, _('Zoom out'), icon=ima.icon('zoom_out'), shortcut=QKeySequence(QKeySequence.ZoomOut), triggered=self.zoom_out)
        zoom_reset_action = create_action(self, _('Zoom reset'), shortcut=QKeySequence('Ctrl+0'), triggered=self.zoom_reset)
        writer = self.writer_docstring
        self.docstring_action = create_action(self, _('Generate docstring'), shortcut=self.get_shortcut('docstring'), triggered=writer.write_docstring_at_first_line_of_function)
        formatter = self.get_conf(('provider_configuration', 'lsp', 'values', 'formatting'), default='', section='completions')
        self.format_action = create_action(self, _('Format file or selection with {0}').format(formatter.capitalize()), shortcut=self.get_shortcut('autoformatting'), triggered=self.format_document_or_range)
        self.format_action.setEnabled(False)
        self.menu = QMenu(self)
        actions_1 = [self.gotodef_action, self.inspect_current_object_action, None, self.undo_action, self.redo_action, None, self.cut_action, self.copy_action, self.paste_action, selectall_action]
        actions_2 = [None, zoom_in_action, zoom_out_action, zoom_reset_action, None, toggle_comment_action, self.docstring_action, self.format_action]
        if nbformat is not None:
            nb_actions = [self.clear_all_output_action, self.ipynb_convert_action, None]
            actions = actions_1 + nb_actions + actions_2
            add_actions(self.menu, actions)
        else:
            actions = actions_1 + actions_2
            add_actions(self.menu, actions)
        self.readonly_menu = QMenu(self)
        add_actions(self.readonly_menu, (self.copy_action, None, selectall_action, self.gotodef_action))

    def keyReleaseEvent(self, event):
        if False:
            return 10
        'Override Qt method.'
        self.sig_key_released.emit(event)
        key = event.key()
        direction_keys = {Qt.Key_Up, Qt.Key_Left, Qt.Key_Right, Qt.Key_Down}
        if key in direction_keys:
            self.request_cursor_event()
        if key in {Qt.Key_Up, Qt.Key_Down}:
            self.update_decorations_timer.start()
        if event.text():
            if self.timer_syntax_highlight.isActive():
                self.timer_syntax_highlight.stop()
            total_lines = self.get_line_count()
            if total_lines < 1000:
                self.timer_syntax_highlight.setInterval(600)
            elif total_lines < 2000:
                self.timer_syntax_highlight.setInterval(800)
            else:
                self.timer_syntax_highlight.setInterval(1000)
            self.timer_syntax_highlight.start()
        self._restore_editor_cursor_and_selections()
        super(CodeEditor, self).keyReleaseEvent(event)
        event.ignore()

    def event(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Qt method override.'
        if event.type() == QEvent.ShortcutOverride:
            event.ignore()
            return False
        else:
            return super(CodeEditor, self).event(event)

    def _handle_keypress_event(self, event):
        if False:
            while True:
                i = 10
        'Handle keypress events.'
        TextEditBaseWidget.keyPressEvent(self, event)
        text = to_text_string(event.text())
        if text:
            if parse(QT_VERSION) < parse('5.15') or os.name == 'nt' or sys.platform == 'darwin':
                cursor = self.textCursor()
                cursor.setPosition(cursor.position())
                self.setTextCursor(cursor)
            self.sig_text_was_inserted.emit()

    def keyPressEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Reimplement Qt method.'
        if self.completions_hint_after_ms > 0:
            self._completions_hint_idle = False
            self._timer_completions_hint.start(self.completions_hint_after_ms)
        else:
            self._set_completions_hint_idle()
        event.ignore()
        self.sig_key_pressed.emit(event)
        self._last_pressed_key = key = event.key()
        self._last_key_pressed_text = text = to_text_string(event.text())
        has_selection = self.has_selected_text()
        ctrl = event.modifiers() & Qt.ControlModifier
        shift = event.modifiers() & Qt.ShiftModifier
        if text:
            self.clear_occurrences()
        if key in {Qt.Key_Up, Qt.Key_Left, Qt.Key_Right, Qt.Key_Down}:
            self.hide_tooltip()
        if event.isAccepted():
            return
        if key in [Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta, Qt.KeypadModifier]:
            if ctrl:
                pos = self.mapFromGlobal(QCursor.pos())
                pos = self.calculate_real_position_from_global(pos)
                if self._handle_goto_uri_event(pos):
                    event.accept()
                    return
                if self._handle_goto_definition_event(pos):
                    event.accept()
                    return
            return
        operators = {'+', '-', '*', '**', '/', '//', '%', '@', '<<', '>>', '&', '|', '^', '~', '<', '>', '<=', '>=', '==', '!='}
        delimiters = {',', ':', ';', '@', '=', '->', '+=', '-=', '*=', '/=', '//=', '%=', '@=', '&=', '|=', '^=', '>>=', '<<=', '**='}
        if text not in self.auto_completion_characters:
            if text in operators or text in delimiters:
                self.completion_widget.hide()
        if key in (Qt.Key_Enter, Qt.Key_Return):
            if not shift and (not ctrl):
                if self.add_colons_enabled and self.is_python_like() and self.autoinsert_colons():
                    self.textCursor().beginEditBlock()
                    self.insert_text(':' + self.get_line_separator())
                    if self.strip_trailing_spaces_on_modify:
                        self.fix_and_strip_indent()
                    else:
                        self.fix_indent()
                    self.textCursor().endEditBlock()
                elif self.is_completion_widget_visible():
                    self.select_completion_list()
                else:
                    self.textCursor().beginEditBlock()
                    cur_indent = self.get_block_indentation(self.textCursor().blockNumber())
                    self._handle_keypress_event(event)
                    cmt_or_str_cursor = self.in_comment_or_string()
                    cursor = self.textCursor()
                    cursor.setPosition(cursor.block().position(), QTextCursor.KeepAnchor)
                    cmt_or_str_line_begin = self.in_comment_or_string(cursor=cursor)
                    cmt_or_str = cmt_or_str_cursor and cmt_or_str_line_begin
                    if self.strip_trailing_spaces_on_modify:
                        self.fix_and_strip_indent(comment_or_string=cmt_or_str, cur_indent=cur_indent)
                    else:
                        self.fix_indent(comment_or_string=cmt_or_str, cur_indent=cur_indent)
                    self.textCursor().endEditBlock()
        elif key == Qt.Key_Insert and (not shift) and (not ctrl):
            self.setOverwriteMode(not self.overwriteMode())
        elif key == Qt.Key_Backspace and (not shift) and (not ctrl):
            if has_selection or not self.intelligent_backspace:
                self._handle_keypress_event(event)
            else:
                leading_text = self.get_text('sol', 'cursor')
                leading_length = len(leading_text)
                trailing_spaces = leading_length - len(leading_text.rstrip())
                trailing_text = self.get_text('cursor', 'eol')
                matches = ('()', '[]', '{}', "''", '""')
                if not leading_text.strip() and leading_length > len(self.indent_chars):
                    if leading_length % len(self.indent_chars) == 0:
                        self.unindent()
                    else:
                        self._handle_keypress_event(event)
                elif trailing_spaces and (not trailing_text.strip()):
                    self.remove_suffix(leading_text[-trailing_spaces:])
                elif leading_text and trailing_text and (leading_text[-1] + trailing_text[0] in matches):
                    cursor = self.textCursor()
                    cursor.movePosition(QTextCursor.PreviousCharacter)
                    cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor, 2)
                    cursor.removeSelectedText()
                else:
                    self._handle_keypress_event(event)
        elif key == Qt.Key_Home:
            self.stdkey_home(shift, ctrl)
        elif key == Qt.Key_End:
            self.stdkey_end(shift, ctrl)
        elif text in self.auto_completion_characters and self.automatic_completions:
            self.insert_text(text)
            if text == '.':
                if not self.in_comment_or_string():
                    text = self.get_text('sol', 'cursor')
                    last_obj = getobj(text)
                    prev_char = text[-2] if len(text) > 1 else ''
                    if prev_char in {')', ']', '}'} or (last_obj and (not last_obj.isdigit())):
                        self.do_completion(automatic=True)
            else:
                self.do_completion(automatic=True)
        elif text in self.signature_completion_characters and (not self.has_selected_text()):
            self.insert_text(text)
            self.request_signature()
        elif key == Qt.Key_Colon and (not has_selection) and self.auto_unindent_enabled:
            leading_text = self.get_text('sol', 'cursor')
            if leading_text.lstrip() in ('else', 'finally'):
                ind = lambda txt: len(txt) - len(txt.lstrip())
                prevtxt = to_text_string(self.textCursor().block().previous().text())
                if self.language == 'Python':
                    prevtxt = prevtxt.rstrip()
                if ind(leading_text) == ind(prevtxt):
                    self.unindent(force=True)
            self._handle_keypress_event(event)
        elif key == Qt.Key_Space and (not shift) and (not ctrl) and (not has_selection) and self.auto_unindent_enabled:
            self.completion_widget.hide()
            leading_text = self.get_text('sol', 'cursor')
            if leading_text.lstrip() in ('elif', 'except'):
                ind = lambda txt: len(txt) - len(txt.lstrip())
                prevtxt = to_text_string(self.textCursor().block().previous().text())
                if self.language == 'Python':
                    prevtxt = prevtxt.rstrip()
                if ind(leading_text) == ind(prevtxt):
                    self.unindent(force=True)
            self._handle_keypress_event(event)
        elif key == Qt.Key_Tab and (not ctrl):
            if not has_selection and (not self.tab_mode):
                self.intelligent_tab()
            else:
                self.indent_or_replace()
        elif key == Qt.Key_Backtab and (not ctrl):
            if not has_selection and (not self.tab_mode):
                self.intelligent_backtab()
            else:
                self.unindent()
            event.accept()
        elif not event.isAccepted():
            self._handle_keypress_event(event)
        if not event.modifiers():
            event.accept()

    def do_automatic_completions(self):
        if False:
            print('Hello World!')
        'Perform on the fly completions.'
        if not self.automatic_completions:
            return
        cursor = self.textCursor()
        pos = cursor.position()
        cursor.select(QTextCursor.WordUnderCursor)
        text = to_text_string(cursor.selectedText())
        key = self._last_pressed_key
        if key is not None:
            if key in [Qt.Key_Return, Qt.Key_Escape, Qt.Key_Tab, Qt.Key_Backtab, Qt.Key_Space]:
                self._last_pressed_key = None
                return
        if key == Qt.Key_Backspace:
            cursor.setPosition(max(0, pos - 1), QTextCursor.MoveAnchor)
            cursor.select(QTextCursor.WordUnderCursor)
            prev_text = to_text_string(cursor.selectedText())
            cursor.setPosition(max(0, pos - 1), QTextCursor.MoveAnchor)
            cursor.setPosition(pos, QTextCursor.KeepAnchor)
            prev_char = cursor.selectedText()
            if prev_text == '' or prev_char in (u'\u2029', ' ', '\t'):
                return
        if text == '':
            cursor.setPosition(max(0, pos - 1), QTextCursor.MoveAnchor)
            cursor.select(QTextCursor.WordUnderCursor)
            text = to_text_string(cursor.selectedText())
            if text != '.':
                text = ''
        if text.startswith((')', ']', '}')):
            cursor.setPosition(pos - 1, QTextCursor.MoveAnchor)
            cursor.select(QTextCursor.WordUnderCursor)
            text = to_text_string(cursor.selectedText())
        is_backspace = self.is_completion_widget_visible() and key == Qt.Key_Backspace
        if len(text) >= self.automatic_completions_after_chars and self._last_key_pressed_text or is_backspace:
            if not self.in_comment_or_string():
                if text.isalpha() or text.isalnum() or '_' in text or ('.' in text):
                    self.do_completion(automatic=True)
                    self._last_key_pressed_text = ''
                    self._last_pressed_key = None

    def fix_and_strip_indent(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Automatically fix indent and strip previous automatic indent.\n\n        args and kwargs are forwarded to self.fix_indent\n        '
        cursor_before = self.textCursor().position()
        if cursor_before > 0:
            self.last_change_position = cursor_before - 1
        self.fix_indent(*args, **kwargs)
        cursor_after = self.textCursor().position()
        nspaces_removed = self.strip_trailing_spaces()
        self.last_auto_indent = (cursor_before - nspaces_removed, cursor_after - nspaces_removed)

    def run_pygments_highlighter(self):
        if False:
            print('Hello World!')
        'Run pygments highlighter.'
        if isinstance(self.highlighter, sh.PygmentsSH):
            self.highlighter.make_charlist()

    def get_pattern_at(self, coordinates):
        if False:
            print('Hello World!')
        '\n        Return key, text and cursor for pattern (if found at coordinates).\n        '
        return self.get_pattern_cursor_at(self.highlighter.patterns, coordinates)

    def get_pattern_cursor_at(self, pattern, coordinates):
        if False:
            while True:
                i = 10
        '\n        Find pattern located at the line where the coordinate is located.\n\n        This returns the actual match and the cursor that selects the text.\n        '
        (cursor, key, text) = (None, None, None)
        break_loop = False
        line = self.get_line_at(coordinates)
        for match in pattern.finditer(line):
            for (key, value) in list(match.groupdict().items()):
                if value:
                    (start, end) = sh.get_span(match)
                    cursor = self.cursorForPosition(coordinates)
                    cursor.movePosition(QTextCursor.StartOfBlock)
                    line_start_position = cursor.position()
                    cursor.setPosition(line_start_position + start, cursor.MoveAnchor)
                    start_rect = self.cursorRect(cursor)
                    cursor.setPosition(line_start_position + end, cursor.MoveAnchor)
                    end_rect = self.cursorRect(cursor)
                    bounding_rect = start_rect.united(end_rect)
                    if bounding_rect.contains(coordinates):
                        text = line[start:end]
                        cursor.setPosition(line_start_position + start, cursor.KeepAnchor)
                        break_loop = True
                        break
            if break_loop:
                break
        return (key, text, cursor)

    def _preprocess_file_uri(self, uri):
        if False:
            while True:
                i = 10
        'Format uri to conform to absolute or relative file paths.'
        fname = uri.replace('file://', '')
        if fname[-1] == '/':
            fname = fname[:-1]
        if fname.startswith('^/'):
            if self.current_project_path is not None:
                fname = osp.join(self.current_project_path, fname[2:])
            else:
                fname = fname.replace('^/', '~/')
        if fname.startswith('~/'):
            fname = osp.expanduser(fname)
        dirname = osp.dirname(osp.abspath(self.filename))
        if osp.isdir(dirname):
            if not osp.isfile(fname):
                fname = osp.join(dirname, fname)
        self.sig_file_uri_preprocessed.emit(fname)
        return fname

    def _handle_goto_definition_event(self, pos):
        if False:
            return 10
        'Check if goto definition can be applied and apply highlight.'
        text = self.get_word_at(pos)
        if text and (not sourcecode.is_keyword(to_text_string(text))):
            if not self.__cursor_changed:
                QApplication.setOverrideCursor(QCursor(Qt.PointingHandCursor))
                self.__cursor_changed = True
            cursor = self.cursorForPosition(pos)
            cursor.select(QTextCursor.WordUnderCursor)
            self.clear_extra_selections('ctrl_click')
            self.highlight_selection('ctrl_click', cursor, foreground_color=self.ctrl_click_color, underline_color=self.ctrl_click_color, underline_style=QTextCharFormat.SingleUnderline)
            return True
        else:
            return False

    def _handle_goto_uri_event(self, pos):
        if False:
            i = 10
            return i + 15
        'Check if go to uri can be applied and apply highlight.'
        (key, pattern_text, cursor) = self.get_pattern_at(pos)
        if key and pattern_text and cursor:
            self._last_hover_pattern_key = key
            self._last_hover_pattern_text = pattern_text
            color = self.ctrl_click_color
            if key in ['file']:
                fname = self._preprocess_file_uri(pattern_text)
                if not osp.isfile(fname):
                    color = QColor(SpyderPalette.COLOR_ERROR_2)
            self.clear_extra_selections('ctrl_click')
            self.highlight_selection('ctrl_click', cursor, foreground_color=color, underline_color=color, underline_style=QTextCharFormat.SingleUnderline)
            if not self.__cursor_changed:
                QApplication.setOverrideCursor(QCursor(Qt.PointingHandCursor))
                self.__cursor_changed = True
            self.sig_uri_found.emit(pattern_text)
            return True
        else:
            self._last_hover_pattern_key = key
            self._last_hover_pattern_text = pattern_text
            return False

    def go_to_uri_from_cursor(self, uri):
        if False:
            i = 10
            return i + 15
        'Go to url from cursor and defined hover patterns.'
        key = self._last_hover_pattern_key
        full_uri = uri
        if key in ['file']:
            fname = self._preprocess_file_uri(uri)
            if osp.isfile(fname) and encoding.is_text_file(fname):
                self.go_to_definition.emit(fname, 0, 0)
            else:
                fname = file_uri(fname)
                start_file(fname)
        elif key in ['mail', 'url']:
            if '@' in uri and (not uri.startswith('mailto:')):
                full_uri = 'mailto:' + uri
            quri = QUrl(full_uri)
            QDesktopServices.openUrl(quri)
        elif key in ['issue']:
            repo_url = uri.replace('#', '/issues/')
            if uri.startswith(('gh-', 'bb-', 'gl-')):
                number = uri[3:]
                remotes = get_git_remotes(self.filename)
                remote = remotes.get('upstream', remotes.get('origin'))
                if remote:
                    full_uri = remote_to_url(remote) + '/issues/' + number
                else:
                    full_uri = None
            elif uri.startswith('gh:') or ':' not in uri:
                repo_and_issue = repo_url
                if uri.startswith('gh:'):
                    repo_and_issue = repo_url[3:]
                full_uri = 'https://github.com/' + repo_and_issue
            elif uri.startswith('gl:'):
                full_uri = 'https://gitlab.com/' + repo_url[3:]
            elif uri.startswith('bb:'):
                full_uri = 'https://bitbucket.org/' + repo_url[3:]
            if full_uri:
                quri = QUrl(full_uri)
                QDesktopServices.openUrl(quri)
            else:
                QMessageBox.information(self, _('Information'), _('This file is not part of a local repository or upstream/origin remotes are not defined!'), QMessageBox.Ok)
        self.hide_tooltip()
        return full_uri

    def line_range(self, position):
        if False:
            i = 10
            return i + 15
        '\n        Get line range from position.\n        '
        if position is None:
            return None
        if position >= self.document().characterCount():
            return None
        cursor = self.textCursor()
        cursor.setPosition(position)
        line_range = (cursor.block().position(), cursor.block().position() + cursor.block().length() - 1)
        return line_range

    def strip_trailing_spaces(self):
        if False:
            return 10
        '\n        Strip trailing spaces if needed.\n\n        Remove trailing whitespace on leaving a non-string line containing it.\n        Return the number of removed spaces.\n        '
        if not running_under_pytest():
            if not self.hasFocus():
                return 0
        current_position = self.textCursor().position()
        last_position = self.last_position
        self.last_position = current_position
        if self.skip_rstrip:
            return 0
        line_range = self.line_range(last_position)
        if line_range is None:
            return 0

        def pos_in_line(pos):
            if False:
                while True:
                    i = 10
            'Check if pos is in last line.'
            if pos is None:
                return False
            return line_range[0] <= pos <= line_range[1]
        if pos_in_line(current_position):
            return 0
        cursor = self.textCursor()
        cursor.setPosition(line_range[1])
        if not self.strip_trailing_spaces_on_modify or self.in_string(cursor=cursor):
            if self.last_auto_indent is None:
                return 0
            elif self.last_auto_indent != self.line_range(self.last_auto_indent[0]):
                self.last_auto_indent = None
                return 0
            line_range = self.last_auto_indent
            self.last_auto_indent = None
        elif not pos_in_line(self.last_change_position):
            return 0
        cursor.setPosition(line_range[0])
        cursor.setPosition(line_range[1], QTextCursor.KeepAnchor)
        text = cursor.selectedText()
        strip = text.rstrip()
        N_strip = qstring_length(text[len(strip):])
        if N_strip > 0:
            cursor.setPosition(line_range[1] - N_strip)
            cursor.setPosition(line_range[1], QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            self.last_change_position = line_range[1]
            self.last_position = self.textCursor().position()
            return N_strip
        return 0

    def move_line_up(self):
        if False:
            return 10
        'Move up current line or selected text'
        self.__move_line_or_selection(after_current_line=False)

    def move_line_down(self):
        if False:
            print('Hello World!')
        'Move down current line or selected text'
        self.__move_line_or_selection(after_current_line=True)

    def __move_line_or_selection(self, after_current_line=True):
        if False:
            return 10
        cursor = self.textCursor()
        folding_panel = self.panels.get('FoldingPanel')
        fold_start_line = cursor.blockNumber() + 1
        block = cursor.block().next()
        if fold_start_line in folding_panel.folding_status:
            fold_status = folding_panel.folding_status[fold_start_line]
            if fold_status:
                folding_panel.toggle_fold_trigger(block)
        if after_current_line:
            fold_start_line = cursor.blockNumber() + 2
            block = cursor.block().next().next()
            if fold_start_line in folding_panel.folding_status:
                fold_status = folding_panel.folding_status[fold_start_line]
                if fold_status:
                    folding_panel.toggle_fold_trigger(block)
        else:
            block = cursor.block()
            offset = 0
            if self.has_selected_text():
                ((selection_start, _), selection_end) = self.get_selection_start_end()
                if selection_end != selection_start:
                    offset = 1
            fold_start_line = block.blockNumber() - 1 - offset
            enclosing_regions = sorted(list(folding_panel.current_tree[fold_start_line]))
            folding_status = folding_panel.folding_status
            if len(enclosing_regions) > 0:
                for region in enclosing_regions:
                    fold_start_line = region.begin
                    block = self.document().findBlockByNumber(fold_start_line)
                    if fold_start_line in folding_status:
                        fold_status = folding_status[fold_start_line]
                        if fold_status:
                            folding_panel.toggle_fold_trigger(block)
        self._TextEditBaseWidget__move_line_or_selection(after_current_line=after_current_line)

    def mouseMoveEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Underline words when pressing <CONTROL>'
        self._timer_mouse_moving.start()
        pos = event.pos()
        self._last_point = pos
        alt = event.modifiers() & Qt.AltModifier
        ctrl = event.modifiers() & Qt.ControlModifier
        if alt:
            self.sig_alt_mouse_moved.emit(event)
            event.accept()
            return
        if ctrl:
            if self._handle_goto_uri_event(pos):
                event.accept()
                return
        if self.has_selected_text():
            TextEditBaseWidget.mouseMoveEvent(self, event)
            return
        if self.go_to_definition_enabled and ctrl:
            if self._handle_goto_definition_event(pos):
                event.accept()
                return
        if self.__cursor_changed:
            self._restore_editor_cursor_and_selections()
        elif not self._should_display_hover(pos) and (not self.is_completion_widget_visible()):
            self.hide_tooltip()
        TextEditBaseWidget.mouseMoveEvent(self, event)

    def setPlainText(self, txt):
        if False:
            return 10
        '\n        Extends setPlainText to emit the new_text_set signal.\n\n        :param txt: The new text to set.\n        :param mime_type: Associated mimetype. Setting the mime will update the\n                          pygments lexer.\n        :param encoding: text encoding\n        '
        super(CodeEditor, self).setPlainText(txt)
        self.new_text_set.emit()

    def focusOutEvent(self, event):
        if False:
            while True:
                i = 10
        'Extend Qt method'
        self.sig_focus_changed.emit()
        self._restore_editor_cursor_and_selections()
        super(CodeEditor, self).focusOutEvent(event)

    def focusInEvent(self, event):
        if False:
            i = 10
            return i + 15
        formatting_enabled = getattr(self, 'formatting_enabled', False)
        self.sig_refresh_formatting.emit(formatting_enabled)
        super(CodeEditor, self).focusInEvent(event)

    def leaveEvent(self, event):
        if False:
            print('Hello World!')
        'Extend Qt method'
        self.sig_leave_out.emit()
        self._restore_editor_cursor_and_selections()
        TextEditBaseWidget.leaveEvent(self, event)

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Override Qt method.'
        self.hide_tooltip()
        ctrl = event.modifiers() & Qt.ControlModifier
        alt = event.modifiers() & Qt.AltModifier
        pos = event.pos()
        self._mouse_left_button_pressed = event.button() == Qt.LeftButton
        if event.button() == Qt.LeftButton and ctrl:
            TextEditBaseWidget.mousePressEvent(self, event)
            cursor = self.cursorForPosition(pos)
            uri = self._last_hover_pattern_text
            if uri:
                self.go_to_uri_from_cursor(uri)
            else:
                self.go_to_definition_from_cursor(cursor)
        elif event.button() == Qt.LeftButton and alt:
            self.sig_alt_left_mouse_pressed.emit(event)
        else:
            TextEditBaseWidget.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        if False:
            print('Hello World!')
        'Override Qt method.'
        if event.button() == Qt.LeftButton:
            self._mouse_left_button_pressed = False
        self.request_cursor_event()
        TextEditBaseWidget.mouseReleaseEvent(self, event)

    def contextMenuEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Reimplement Qt method'
        nonempty_selection = self.has_selected_text()
        self.copy_action.setEnabled(nonempty_selection)
        self.cut_action.setEnabled(nonempty_selection)
        self.clear_all_output_action.setVisible(self.is_json() and nbformat is not None)
        self.ipynb_convert_action.setVisible(self.is_json() and nbformat is not None)
        self.gotodef_action.setVisible(self.go_to_definition_enabled)
        formatter = self.get_conf(('provider_configuration', 'lsp', 'values', 'formatting'), default='', section='completions')
        self.format_action.setText(_('Format file or selection with {0}').format(formatter.capitalize()))
        writer = self.writer_docstring
        writer.line_number_cursor = self.get_line_number_at(event.pos())
        result = writer.get_function_definition_from_first_line()
        if result:
            self.docstring_action.setEnabled(True)
        else:
            self.docstring_action.setEnabled(False)
        cursor = self.textCursor()
        text = to_text_string(cursor.selectedText())
        if len(text) == 0:
            cursor.select(QTextCursor.WordUnderCursor)
            text = to_text_string(cursor.selectedText())
        self.undo_action.setEnabled(self.document().isUndoAvailable())
        self.redo_action.setEnabled(self.document().isRedoAvailable())
        menu = self.menu
        if self.isReadOnly():
            menu = self.readonly_menu
        menu.popup(event.globalPos())
        event.accept()

    def _restore_editor_cursor_and_selections(self):
        if False:
            while True:
                i = 10
        'Restore the cursor and extra selections of this code editor.'
        if self.__cursor_changed:
            self.__cursor_changed = False
            QApplication.restoreOverrideCursor()
            self.clear_extra_selections('ctrl_click')
            self._last_hover_pattern_key = None
            self._last_hover_pattern_text = None

    def dragEnterEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reimplemented Qt method.\n\n        Inform Qt about the types of data that the widget accepts.\n        '
        logger.debug('dragEnterEvent was received')
        all_urls = mimedata2url(event.mimeData())
        if all_urls:
            logger.debug('Let the parent widget handle this dragEnterEvent')
            event.ignore()
        else:
            logger.debug('Call TextEditBaseWidget dragEnterEvent method')
            TextEditBaseWidget.dragEnterEvent(self, event)

    def dropEvent(self, event):
        if False:
            return 10
        '\n        Reimplemented Qt method.\n\n        Unpack dropped data and handle it.\n        '
        logger.debug('dropEvent was received')
        if mimedata2url(event.mimeData()):
            logger.debug('Let the parent widget handle this')
            event.ignore()
        else:
            logger.debug('Call TextEditBaseWidget dropEvent method')
            TextEditBaseWidget.dropEvent(self, event)

    def paintEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Overrides paint event to update the list of visible blocks'
        self.update_visible_blocks(event)
        TextEditBaseWidget.paintEvent(self, event)
        self.painted.emit(event)

    def update_visible_blocks(self, event):
        if False:
            print('Hello World!')
        'Update the list of visible blocks/lines position'
        self.__visible_blocks[:] = []
        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        ebottom_bottom = self.height()
        while block.isValid():
            visible = bottom <= ebottom_bottom
            if not visible:
                break
            if block.isVisible():
                self.__visible_blocks.append((top, blockNumber + 1, block))
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            blockNumber = block.blockNumber()

    def _draw_editor_cell_divider(self):
        if False:
            for i in range(10):
                print('nop')
        'Draw a line on top of a define cell'
        if self.supported_cell_language:
            cell_line_color = self.comment_color
            painter = QPainter(self.viewport())
            pen = painter.pen()
            pen.setStyle(Qt.SolidLine)
            pen.setBrush(cell_line_color)
            painter.setPen(pen)
            for (top, line_number, block) in self.visible_blocks:
                if is_cell_header(block):
                    painter.drawLine(0, top, self.width(), top)

    @property
    def visible_blocks(self):
        if False:
            while True:
                i = 10
        '\n        Returns the list of visible blocks.\n\n        Each element in the list is a tuple made up of the line top position,\n        the line number (already 1 based), and the QTextBlock itself.\n\n        :return: A list of tuple(top position, line number, block)\n        :rtype: List of tuple(int, int, QtGui.QTextBlock)\n        '
        return self.__visible_blocks

    def is_editor(self):
        if False:
            i = 10
            return i + 15
        return True

    def popup_docstring(self, prev_text, prev_pos):
        if False:
            while True:
                i = 10
        'Show the menu for generating docstring.'
        line_text = self.textCursor().block().text()
        if line_text != prev_text:
            return
        if prev_pos != self.textCursor().position():
            return
        writer = self.writer_docstring
        if writer.get_function_definition_from_below_last_line():
            point = self.cursorRect().bottomRight()
            point = self.calculate_real_position(point)
            point = self.mapToGlobal(point)
            self.menu_docstring = QMenuOnlyForEnter(self)
            self.docstring_action = create_action(self, _('Generate docstring'), icon=ima.icon('TextFileIcon'), triggered=writer.write_docstring)
            self.menu_docstring.addAction(self.docstring_action)
            self.menu_docstring.setActiveAction(self.docstring_action)
            self.menu_docstring.popup(point)

    def delayed_popup_docstring(self):
        if False:
            return 10
        "Show context menu for docstring.\n\n        This method is called after typing '''. After typing ''', this function\n        waits 300ms. If there was no input for 300ms, show the context menu.\n        "
        line_text = self.textCursor().block().text()
        pos = self.textCursor().position()
        timer = QTimer()
        timer.singleShot(300, lambda : self.popup_docstring(line_text, pos))

    def set_current_project_path(self, root_path=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the current active project root path.\n\n        Parameters\n        ----------\n        root_path: str or None, optional\n            Path to current project root path. Default is None.\n        '
        self.current_project_path = root_path

    def count_leading_empty_lines(self, cell):
        if False:
            i = 10
            return i + 15
        'Count the number of leading empty cells.'
        lines = cell.splitlines(keepends=True)
        if not lines:
            return 0
        for (i, line) in enumerate(lines):
            if line and (not line.isspace()):
                return i
        return len(lines)

    def ipython_to_python(self, code):
        if False:
            return 10
        'Transform IPython code to python code.'
        tm = TransformerManager()
        number_empty_lines = self.count_leading_empty_lines(code)
        try:
            code = tm.transform_cell(code)
        except SyntaxError:
            return code
        return '\n' * number_empty_lines + code

    def is_letter_or_number(self, char):
        if False:
            return 10
        '\n        Returns whether the specified unicode character is a letter or a\n        number.\n        '
        cat = category(char)
        return cat.startswith('L') or cat.startswith('N')

class TestWidget(QSplitter):

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        QSplitter.__init__(self, parent)
        self.editor = CodeEditor(self)
        self.editor.setup_editor(linenumbers=True, markers=True, tab_mode=False, font=QFont('Courier New', 10), show_blanks=True, color_scheme='Zenburn')
        self.addWidget(self.editor)
        self.setWindowIcon(ima.icon('spyder'))

    def load(self, filename):
        if False:
            return 10
        self.editor.set_text_from_file(filename)
        self.setWindowTitle('%s - %s (%s)' % (_('Editor'), osp.basename(filename), osp.dirname(filename)))
        self.editor.hide_tooltip()

def test(fname):
    if False:
        i = 10
        return i + 15
    from spyder.utils.qthelpers import qapplication
    app = qapplication(test_time=5)
    win = TestWidget(None)
    win.show()
    win.load(fname)
    win.resize(900, 700)
    sys.exit(app.exec_())
if __name__ == '__main__':
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = __file__
    test(fname)