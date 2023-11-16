"""An abstract base class for console-type widgets."""
from functools import partial
import os
import os.path
import re
import sys
from textwrap import dedent
import time
from unicodedata import category
import webbrowser
from qtpy import QT6
from qtpy import QtCore, QtGui, QtPrintSupport, QtWidgets
from qtconsole.rich_text import HtmlExporter
from qtconsole.util import MetaQObjectHasTraits, get_font, superQ
from traitlets.config.configurable import LoggingConfigurable
from traitlets import Bool, Enum, Integer, Unicode
from .ansi_code_processor import QtAnsiCodeProcessor
from .completion_widget import CompletionWidget
from .completion_html import CompletionHtml
from .completion_plain import CompletionPlain
from .kill_ring import QtKillRing
from .util import columnize

def is_letter_or_number(char):
    if False:
        i = 10
        return i + 15
    ' Returns whether the specified unicode character is a letter or a number.\n    '
    cat = category(char)
    return cat.startswith('L') or cat.startswith('N')

def is_whitespace(char):
    if False:
        print('Hello World!')
    'Check whether a given char counts as white space.'
    return category(char).startswith('Z')

class ConsoleWidget(MetaQObjectHasTraits('NewBase', (LoggingConfigurable, superQ(QtWidgets.QWidget)), {})):
    """ An abstract base class for console-type widgets. This class has
        functionality for:

            * Maintaining a prompt and editing region
            * Providing the traditional Unix-style console keyboard shortcuts
            * Performing tab completion
            * Paging text
            * Handling ANSI escape codes

        ConsoleWidget also provides a number of utility methods that will be
        convenient to implementors of a console-style widget.
    """
    ansi_codes = Bool(True, config=True, help='Whether to process ANSI escape codes.')
    buffer_size = Integer(500, config=True, help='\n        The maximum number of lines of text before truncation. Specifying a\n        non-positive number disables text truncation (not recommended).\n        ')
    execute_on_complete_input = Bool(True, config=True, help='Whether to automatically execute on syntactically complete input.\n\n        If False, Shift-Enter is required to submit each execution.\n        Disabling this is mainly useful for non-Python kernels,\n        where the completion check would be wrong.\n        ')
    gui_completion = Enum(['plain', 'droplist', 'ncurses'], config=True, default_value='ncurses', help="\n                    The type of completer to use. Valid values are:\n\n                    'plain'   : Show the available completion as a text list\n                                Below the editing area.\n                    'droplist': Show the completion in a drop down list navigable\n                                by the arrow keys, and from which you can select\n                                completion by pressing Return.\n                    'ncurses' : Show the completion as a text list which is navigable by\n                                `tab` and arrow keys.\n                    ")
    gui_completion_height = Integer(0, config=True, help="\n        Set Height for completion.\n\n        'droplist'\n            Height in pixels.\n        'ncurses'\n            Maximum number of rows.\n        ")
    kind = Enum(['plain', 'rich'], default_value='plain', config=True, help="\n        The type of underlying text widget to use. Valid values are 'plain',\n        which specifies a QPlainTextEdit, and 'rich', which specifies a\n        QTextEdit.\n        ")
    paging = Enum(['inside', 'hsplit', 'vsplit', 'custom', 'none'], default_value='inside', config=True, help="\n        The type of paging to use. Valid values are:\n\n        'inside'\n           The widget pages like a traditional terminal.\n        'hsplit'\n           When paging is requested, the widget is split horizontally. The top\n           pane contains the console, and the bottom pane contains the paged text.\n        'vsplit'\n           Similar to 'hsplit', except that a vertical splitter is used.\n        'custom'\n           No action is taken by the widget beyond emitting a\n           'custom_page_requested(str)' signal.\n        'none'\n           The text is written directly to the console.\n        ")
    scrollbar_visibility = Bool(True, config=True, help='The visibility of the scrollar. If False then the scrollbar will be\n        invisible.')
    font_family = Unicode(config=True, help='The font family to use for the console.\n        On OSX this defaults to Monaco, on Windows the default is\n        Consolas with fallback of Courier, and on other platforms\n        the default is Monospace.\n        ')

    def _font_family_default(self):
        if False:
            print('Hello World!')
        if sys.platform == 'win32':
            return 'Consolas'
        elif sys.platform == 'darwin':
            return 'Monaco'
        else:
            return 'Monospace'
    font_size = Integer(config=True, help='The font size. If unconfigured, Qt will be entrusted\n        with the size of the font.\n        ')
    console_width = Integer(81, config=True, help='The width of the console at start time in number\n        of characters (will double with `hsplit` paging)\n        ')
    console_height = Integer(25, config=True, help='The height of the console at start time in number\n        of characters (will double with `vsplit` paging)\n        ')
    override_shortcuts = Bool(False)
    custom_control = None
    custom_page_control = None
    copy_available = QtCore.Signal(bool)
    redo_available = QtCore.Signal(bool)
    undo_available = QtCore.Signal(bool)
    custom_page_requested = QtCore.Signal(object)
    font_changed = QtCore.Signal(QtGui.QFont)
    _control = None
    _page_control = None
    _splitter = None
    _ctrl_down_remap = {QtCore.Qt.Key_B: QtCore.Qt.Key_Left, QtCore.Qt.Key_F: QtCore.Qt.Key_Right, QtCore.Qt.Key_A: QtCore.Qt.Key_Home, QtCore.Qt.Key_P: QtCore.Qt.Key_Up, QtCore.Qt.Key_N: QtCore.Qt.Key_Down, QtCore.Qt.Key_H: QtCore.Qt.Key_Backspace}
    if not sys.platform == 'darwin':
        _ctrl_down_remap[QtCore.Qt.Key_E] = QtCore.Qt.Key_End
    _shortcuts = set(_ctrl_down_remap.keys()) | {QtCore.Qt.Key_C, QtCore.Qt.Key_G, QtCore.Qt.Key_O, QtCore.Qt.Key_V}
    _temp_buffer_filled = False

    def __init__(self, parent=None, **kw):
        if False:
            print('Hello World!')
        ' Create a ConsoleWidget.\n\n        Parameters\n        ----------\n        parent : QWidget, optional [default None]\n            The parent for this widget.\n        '
        super().__init__(**kw)
        if parent:
            self.setParent(parent)
        self._is_complete_msg_id = None
        self._is_complete_timeout = 0.1
        self._is_complete_max_time = None
        self._pager_scroll_events = [QtCore.QEvent.Wheel]
        if hasattr(QtCore.QEvent, 'NativeGesture'):
            self._pager_scroll_events.append(QtCore.QEvent.NativeGesture)
        layout = QtWidgets.QStackedLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._control = self._create_control()
        if self.paging in ('hsplit', 'vsplit'):
            self._splitter = QtWidgets.QSplitter()
            if self.paging == 'hsplit':
                self._splitter.setOrientation(QtCore.Qt.Horizontal)
            else:
                self._splitter.setOrientation(QtCore.Qt.Vertical)
            self._splitter.addWidget(self._control)
            layout.addWidget(self._splitter)
        else:
            layout.addWidget(self._control)
        if self.paging in ('inside', 'hsplit', 'vsplit'):
            self._page_control = self._create_page_control()
            if self._splitter:
                self._page_control.hide()
                self._splitter.addWidget(self._page_control)
            else:
                layout.addWidget(self._page_control)
        self._append_before_prompt_cursor = self._control.textCursor()
        self._ansi_processor = QtAnsiCodeProcessor()
        if self.gui_completion == 'ncurses':
            self._completion_widget = CompletionHtml(self, self.gui_completion_height)
        elif self.gui_completion == 'droplist':
            self._completion_widget = CompletionWidget(self, self.gui_completion_height)
        elif self.gui_completion == 'plain':
            self._completion_widget = CompletionPlain(self)
        self._continuation_prompt = '> '
        self._continuation_prompt_html = None
        self._executing = False
        self._filter_resize = False
        self._html_exporter = HtmlExporter(self._control)
        self._input_buffer_executing = ''
        self._input_buffer_pending = ''
        self._kill_ring = QtKillRing(self._control)
        self._prompt = ''
        self._prompt_html = None
        self._prompt_cursor = self._control.textCursor()
        self._prompt_sep = ''
        self._reading = False
        self._reading_callback = None
        self._tab_width = 4
        self._pending_insert_text = []
        self._pending_text_flush_interval = QtCore.QTimer(self._control)
        self._pending_text_flush_interval.setInterval(100)
        self._pending_text_flush_interval.setSingleShot(True)
        self._pending_text_flush_interval.timeout.connect(self._on_flush_pending_stream_timer)
        self.reset_font()
        action = QtWidgets.QAction('Print', None)
        action.setEnabled(True)
        printkey = QtGui.QKeySequence(QtGui.QKeySequence.Print)
        if printkey.matches('Ctrl+P') and sys.platform != 'darwin':
            printkey = 'Ctrl+Shift+P'
        action.setShortcut(printkey)
        action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        action.triggered.connect(self.print_)
        self.addAction(action)
        self.print_action = action
        action = QtWidgets.QAction('Save as HTML/XML', None)
        action.setShortcut(QtGui.QKeySequence.Save)
        action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        action.triggered.connect(self.export_html)
        self.addAction(action)
        self.export_action = action
        action = QtWidgets.QAction('Select All', None)
        action.setEnabled(True)
        selectall = QtGui.QKeySequence(QtGui.QKeySequence.SelectAll)
        if selectall.matches('Ctrl+A') and sys.platform != 'darwin':
            selectall = 'Ctrl+Shift+A'
        action.setShortcut(selectall)
        action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        action.triggered.connect(self.select_all_smart)
        self.addAction(action)
        self.select_all_action = action
        self.increase_font_size = QtWidgets.QAction('Bigger Font', self, shortcut=QtGui.QKeySequence.ZoomIn, shortcutContext=QtCore.Qt.WidgetWithChildrenShortcut, statusTip='Increase the font size by one point', triggered=self._increase_font_size)
        self.addAction(self.increase_font_size)
        self.decrease_font_size = QtWidgets.QAction('Smaller Font', self, shortcut=QtGui.QKeySequence.ZoomOut, shortcutContext=QtCore.Qt.WidgetWithChildrenShortcut, statusTip='Decrease the font size by one point', triggered=self._decrease_font_size)
        self.addAction(self.decrease_font_size)
        self.reset_font_size = QtWidgets.QAction('Normal Font', self, shortcut='Ctrl+0', shortcutContext=QtCore.Qt.WidgetWithChildrenShortcut, statusTip='Restore the Normal font size', triggered=self.reset_font)
        self.addAction(self.reset_font_size)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if False:
            while True:
                i = 10
        if e.mimeData().hasUrls():
            e.setDropAction(QtCore.Qt.LinkAction)
            e.accept()
        elif e.mimeData().hasText():
            e.setDropAction(QtCore.Qt.CopyAction)
            e.accept()

    def dragMoveEvent(self, e):
        if False:
            i = 10
            return i + 15
        if e.mimeData().hasUrls():
            pass
        elif e.mimeData().hasText():
            cursor = self._control.cursorForPosition(e.pos())
            if self._in_buffer(cursor.position()):
                e.setDropAction(QtCore.Qt.CopyAction)
                self._control.setTextCursor(cursor)
            else:
                e.setDropAction(QtCore.Qt.IgnoreAction)
            e.accept()

    def dropEvent(self, e):
        if False:
            while True:
                i = 10
        if e.mimeData().hasUrls():
            self._keep_cursor_in_buffer()
            cursor = self._control.textCursor()
            filenames = [url.toLocalFile() for url in e.mimeData().urls()]
            text = ', '.join(("'" + f.replace("'", '\'"\'"\'') + "'" for f in filenames))
            self._insert_plain_text_into_buffer(cursor, text)
        elif e.mimeData().hasText():
            cursor = self._control.cursorForPosition(e.pos())
            if self._in_buffer(cursor.position()):
                text = e.mimeData().text()
                self._insert_plain_text_into_buffer(cursor, text)

    def eventFilter(self, obj, event):
        if False:
            print('Hello World!')
        ' Reimplemented to ensure a console-like behavior in the underlying\n            text widgets.\n        '
        etype = event.type()
        if etype == QtCore.QEvent.KeyPress:
            key = event.key()
            if self._control_key_down(event.modifiers()) and key in self._ctrl_down_remap:
                new_event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, self._ctrl_down_remap[key], QtCore.Qt.NoModifier)
                QtWidgets.QApplication.instance().sendEvent(obj, new_event)
                return True
            elif obj == self._control:
                return self._event_filter_console_keypress(event)
            elif obj == self._page_control:
                return self._event_filter_page_keypress(event)
        elif getattr(event, 'button', False) and etype == QtCore.QEvent.MouseButtonRelease and (event.button() == QtCore.Qt.MiddleButton) and (obj == self._control.viewport()):
            cursor = self._control.cursorForPosition(event.pos())
            self._control.setTextCursor(cursor)
            self.paste(QtGui.QClipboard.Selection)
            return True
        elif etype == QtCore.QEvent.Resize and (not self._filter_resize):
            self._filter_resize = True
            QtWidgets.QApplication.instance().sendEvent(obj, event)
            self._adjust_scrollbars()
            self._filter_resize = False
            return True
        elif etype == QtCore.QEvent.ShortcutOverride and self.override_shortcuts and self._control_key_down(event.modifiers()) and (event.key() in self._shortcuts):
            event.accept()
        elif etype in self._pager_scroll_events and obj == self._page_control:
            self._page_control.repaint()
            return True
        elif etype == QtCore.QEvent.MouseMove:
            anchor = self._control.anchorAt(event.pos())
            if QT6:
                pos = event.globalPosition().toPoint()
            else:
                pos = event.globalPos()
            QtWidgets.QToolTip.showText(pos, anchor)
        return super().eventFilter(obj, event)

    def sizeHint(self):
        if False:
            i = 10
            return i + 15
        ' Reimplemented to suggest a size that is 80 characters wide and\n            25 lines high.\n        '
        font_metrics = QtGui.QFontMetrics(self.font)
        margin = (self._control.frameWidth() + self._control.document().documentMargin()) * 2
        style = self.style()
        splitwidth = style.pixelMetric(QtWidgets.QStyle.PM_SplitterWidth)
        width = self._get_font_width() * self.console_width + margin
        width += style.pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
        if self.paging == 'hsplit':
            width = width * 2 + splitwidth
        height = font_metrics.height() * self.console_height + margin
        if self.paging == 'vsplit':
            height = height * 2 + splitwidth
        return QtCore.QSize(int(width), int(height))
    include_other_output = Bool(False, config=True, help='Whether to include output from clients\n        other than this one sharing the same kernel.\n\n        Outputs are not displayed until enter is pressed.\n        ')
    other_output_prefix = Unicode('[remote] ', config=True, help='Prefix to add to outputs coming from clients other than this one.\n\n        Only relevant if include_other_output is True.\n        ')

    def can_copy(self):
        if False:
            for i in range(10):
                print('nop')
        ' Returns whether text can be copied to the clipboard.\n        '
        return self._control.textCursor().hasSelection()

    def can_cut(self):
        if False:
            for i in range(10):
                print('nop')
        ' Returns whether text can be cut to the clipboard.\n        '
        cursor = self._control.textCursor()
        return cursor.hasSelection() and self._in_buffer(cursor.anchor()) and self._in_buffer(cursor.position())

    def can_paste(self):
        if False:
            print('Hello World!')
        ' Returns whether text can be pasted from the clipboard.\n        '
        if self._control.textInteractionFlags() & QtCore.Qt.TextEditable:
            return bool(QtWidgets.QApplication.clipboard().text())
        return False

    def clear(self, keep_input=True):
        if False:
            return 10
        ' Clear the console.\n\n        Parameters\n        ----------\n        keep_input : bool, optional (default True)\n            If set, restores the old input buffer if a new prompt is written.\n        '
        if self._executing:
            self._control.clear()
        else:
            if keep_input:
                input_buffer = self.input_buffer
            self._control.clear()
            self._show_prompt()
            if keep_input:
                self.input_buffer = input_buffer

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        ' Copy the currently selected text to the clipboard.\n        '
        self.layout().currentWidget().copy()

    def copy_anchor(self, anchor):
        if False:
            print('Hello World!')
        ' Copy anchor text to the clipboard\n        '
        QtWidgets.QApplication.clipboard().setText(anchor)

    def cut(self):
        if False:
            i = 10
            return i + 15
        " Copy the currently selected text to the clipboard and delete it\n            if it's inside the input buffer.\n        "
        self.copy()
        if self.can_cut():
            self._control.textCursor().removeSelectedText()

    def _handle_is_complete_reply(self, msg):
        if False:
            return 10
        if msg['parent_header'].get('msg_id', 0) != self._is_complete_msg_id:
            return
        status = msg['content'].get('status', 'complete')
        indent = msg['content'].get('indent', '')
        self._trigger_is_complete_callback(status != 'incomplete', indent)

    def _trigger_is_complete_callback(self, complete=False, indent=''):
        if False:
            for i in range(10):
                print('nop')
        if self._is_complete_msg_id is not None:
            self._is_complete_msg_id = None
            self._is_complete_callback(complete, indent)

    def _register_is_complete_callback(self, source, callback):
        if False:
            for i in range(10):
                print('nop')
        if self._is_complete_msg_id is not None:
            if self._is_complete_max_time < time.time():
                return
            else:
                self._trigger_is_complete_callback()
        self._is_complete_max_time = time.time() + self._is_complete_timeout
        self._is_complete_callback = callback
        self._is_complete_msg_id = self.kernel_client.is_complete(source)

    def execute(self, source=None, hidden=False, interactive=False):
        if False:
            for i in range(10):
                print('nop')
        " Executes source or the input buffer, possibly prompting for more\n        input.\n\n        Parameters\n        ----------\n        source : str, optional\n\n            The source to execute. If not specified, the input buffer will be\n            used. If specified and 'hidden' is False, the input buffer will be\n            replaced with the source before execution.\n\n        hidden : bool, optional (default False)\n\n            If set, no output will be shown and the prompt will not be modified.\n            In other words, it will be completely invisible to the user that\n            an execution has occurred.\n\n        interactive : bool, optional (default False)\n\n            Whether the console is to treat the source as having been manually\n            entered by the user. The effect of this parameter depends on the\n            subclass implementation.\n\n        Raises\n        ------\n        RuntimeError\n            If incomplete input is given and 'hidden' is True. In this case,\n            it is not possible to prompt for more input.\n\n        Returns\n        -------\n        A boolean indicating whether the source was executed.\n        "
        if source is None:
            source = self.input_buffer
        elif not hidden:
            self.input_buffer = source
        if hidden:
            self._execute(source, hidden)
        elif interactive and self.execute_on_complete_input:
            self._register_is_complete_callback(source, partial(self.do_execute, source))
        else:
            self.do_execute(source, True, '')

    def do_execute(self, source, complete, indent):
        if False:
            print('Hello World!')
        if complete:
            self._append_plain_text('\n')
            self._input_buffer_executing = self.input_buffer
            self._executing = True
            self._finalize_input_request()
            self._execute(source, False)
        else:
            cursor = self._get_end_cursor()
            cursor.beginEditBlock()
            try:
                cursor.insertText('\n')
                self._insert_continuation_prompt(cursor, indent)
            finally:
                cursor.endEditBlock()
            self._control.moveCursor(QtGui.QTextCursor.End)

    def export_html(self):
        if False:
            print('Hello World!')
        ' Shows a dialog to export HTML/XML in various formats.\n        '
        self._html_exporter.export()

    def _finalize_input_request(self):
        if False:
            while True:
                i = 10
        '\n        Set the widget to a non-reading state.\n        '
        self._reading = False
        self._prompt_finished()
        self._append_before_prompt_cursor.setPosition(self._get_end_cursor().position())
        self._control.document().setMaximumBlockCount(self.buffer_size)
        self._control.setUndoRedoEnabled(False)

    def _get_input_buffer(self, force=False):
        if False:
            for i in range(10):
                print('nop')
        ' The text that the user has entered entered at the current prompt.\n\n        If the console is currently executing, the text that is executing will\n        always be returned.\n        '
        if self._executing and (not force):
            return self._input_buffer_executing
        cursor = self._get_end_cursor()
        cursor.setPosition(self._prompt_pos, QtGui.QTextCursor.KeepAnchor)
        input_buffer = cursor.selection().toPlainText()
        return input_buffer.replace('\n' + self._continuation_prompt, '\n')

    def _set_input_buffer(self, string):
        if False:
            i = 10
            return i + 15
        ' Sets the text in the input buffer.\n\n        If the console is currently executing, this call has no *immediate*\n        effect. When the execution is finished, the input buffer will be updated\n        appropriately.\n        '
        if self._executing:
            self._input_buffer_pending = string
            return
        cursor = self._get_end_cursor()
        cursor.beginEditBlock()
        cursor.setPosition(self._prompt_pos, QtGui.QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        self._insert_plain_text_into_buffer(self._get_prompt_cursor(), string)
        cursor.endEditBlock()
        self._control.moveCursor(QtGui.QTextCursor.End)
    input_buffer = property(_get_input_buffer, _set_input_buffer)

    def _get_font(self):
        if False:
            i = 10
            return i + 15
        ' The base font being used by the ConsoleWidget.\n        '
        return self._control.document().defaultFont()

    def _get_font_width(self, font=None):
        if False:
            i = 10
            return i + 15
        if font is None:
            font = self.font
        font_metrics = QtGui.QFontMetrics(font)
        if hasattr(font_metrics, 'horizontalAdvance'):
            return font_metrics.horizontalAdvance(' ')
        else:
            return font_metrics.width(' ')

    def _set_font(self, font):
        if False:
            print('Hello World!')
        ' Sets the base font for the ConsoleWidget to the specified QFont.\n        '
        self._control.setTabStopWidth(self.tab_width * self._get_font_width(font))
        self._completion_widget.setFont(font)
        self._control.document().setDefaultFont(font)
        if self._page_control:
            self._page_control.document().setDefaultFont(font)
        self.font_changed.emit(font)
    font = property(_get_font, _set_font)

    def _set_completion_widget(self, gui_completion):
        if False:
            return 10
        ' Set gui completion widget.\n        '
        if gui_completion == 'ncurses':
            self._completion_widget = CompletionHtml(self)
        elif gui_completion == 'droplist':
            self._completion_widget = CompletionWidget(self)
        elif gui_completion == 'plain':
            self._completion_widget = CompletionPlain(self)
        self.gui_completion = gui_completion

    def open_anchor(self, anchor):
        if False:
            while True:
                i = 10
        ' Open selected anchor in the default webbrowser\n        '
        webbrowser.open(anchor)

    def paste(self, mode=QtGui.QClipboard.Clipboard):
        if False:
            print('Hello World!')
        ' Paste the contents of the clipboard into the input region.\n\n        Parameters\n        ----------\n        mode : QClipboard::Mode, optional [default QClipboard::Clipboard]\n\n            Controls which part of the system clipboard is used. This can be\n            used to access the selection clipboard in X11 and the Find buffer\n            in Mac OS. By default, the regular clipboard is used.\n        '
        if self._control.textInteractionFlags() & QtCore.Qt.TextEditable:
            self._keep_cursor_in_buffer()
            cursor = self._control.textCursor()
            text = QtWidgets.QApplication.clipboard().text(mode).rstrip()
            cursor_offset = cursor.position() - self._get_line_start_pos()
            if text.startswith(' ' * cursor_offset):
                text = text[cursor_offset:]
            self._insert_plain_text_into_buffer(cursor, dedent(text))

    def print_(self, printer=None):
        if False:
            return 10
        ' Print the contents of the ConsoleWidget to the specified QPrinter.\n        '
        if not printer:
            printer = QtPrintSupport.QPrinter()
            if QtPrintSupport.QPrintDialog(printer).exec_() != QtPrintSupport.QPrintDialog.Accepted:
                return
        self._control.print_(printer)

    def prompt_to_top(self):
        if False:
            print('Hello World!')
        ' Moves the prompt to the top of the viewport.\n        '
        if not self._executing:
            prompt_cursor = self._get_prompt_cursor()
            if self._get_cursor().blockNumber() < prompt_cursor.blockNumber():
                self._set_cursor(prompt_cursor)
            self._set_top_cursor(prompt_cursor)

    def redo(self):
        if False:
            return 10
        ' Redo the last operation. If there is no operation to redo, nothing\n            happens.\n        '
        self._control.redo()

    def reset_font(self):
        if False:
            i = 10
            return i + 15
        ' Sets the font to the default fixed-width font for this platform.\n        '
        if sys.platform == 'win32':
            fallback = 'Courier'
        elif sys.platform == 'darwin':
            fallback = 'Monaco'
        else:
            fallback = 'Monospace'
        font = get_font(self.font_family, fallback)
        if self.font_size:
            font.setPointSize(self.font_size)
        else:
            font.setPointSize(QtWidgets.QApplication.instance().font().pointSize())
        font.setStyleHint(QtGui.QFont.TypeWriter)
        self._set_font(font)

    def change_font_size(self, delta):
        if False:
            while True:
                i = 10
        'Change the font size by the specified amount (in points).\n        '
        font = self.font
        size = max(font.pointSize() + delta, 1)
        font.setPointSize(size)
        self._set_font(font)

    def _increase_font_size(self):
        if False:
            while True:
                i = 10
        self.change_font_size(1)

    def _decrease_font_size(self):
        if False:
            print('Hello World!')
        self.change_font_size(-1)

    def select_all_smart(self):
        if False:
            print('Hello World!')
        ' Select current cell, or, if already selected, the whole document.\n        '
        c = self._get_cursor()
        sel_range = (c.selectionStart(), c.selectionEnd())
        c.clearSelection()
        c.setPosition(self._get_prompt_cursor().position())
        c.setPosition(self._get_end_pos(), mode=QtGui.QTextCursor.KeepAnchor)
        new_sel_range = (c.selectionStart(), c.selectionEnd())
        if sel_range == new_sel_range:
            self.select_document()
        else:
            self._control.setTextCursor(c)

    def select_document(self):
        if False:
            i = 10
            return i + 15
        ' Selects all the text in the buffer.\n        '
        self._control.selectAll()

    def _get_tab_width(self):
        if False:
            return 10
        ' The width (in terms of space characters) for tab characters.\n        '
        return self._tab_width

    def _set_tab_width(self, tab_width):
        if False:
            i = 10
            return i + 15
        ' Sets the width (in terms of space characters) for tab characters.\n        '
        self._control.setTabStopWidth(tab_width * self._get_font_width())
        self._tab_width = tab_width
    tab_width = property(_get_tab_width, _set_tab_width)

    def undo(self):
        if False:
            print('Hello World!')
        ' Undo the last operation. If there is no operation to undo, nothing\n            happens.\n        '
        self._control.undo()

    def _is_complete(self, source, interactive):
        if False:
            return 10
        " Returns whether 'source' can be executed. When triggered by an\n            Enter/Return key press, 'interactive' is True; otherwise, it is\n            False.\n        "
        raise NotImplementedError

    def _execute(self, source, hidden):
        if False:
            while True:
                i = 10
        " Execute 'source'. If 'hidden', do not show any output.\n        "
        raise NotImplementedError

    def _prompt_started_hook(self):
        if False:
            i = 10
            return i + 15
        ' Called immediately after a new prompt is displayed.\n        '
        pass

    def _prompt_finished_hook(self):
        if False:
            i = 10
            return i + 15
        ' Called immediately after a prompt is finished, i.e. when some input\n            will be processed and a new prompt displayed.\n        '
        pass

    def _up_pressed(self, shift_modifier):
        if False:
            for i in range(10):
                print('nop')
        ' Called when the up key is pressed. Returns whether to continue\n            processing the event.\n        '
        return True

    def _down_pressed(self, shift_modifier):
        if False:
            while True:
                i = 10
        ' Called when the down key is pressed. Returns whether to continue\n            processing the event.\n        '
        return True

    def _tab_pressed(self):
        if False:
            while True:
                i = 10
        ' Called when the tab key is pressed. Returns whether to continue\n            processing the event.\n        '
        return True

    def _append_custom(self, insert, input, before_prompt=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        " A low-level method for appending content to the end of the buffer.\n\n        If 'before_prompt' is enabled, the content will be inserted before the\n        current prompt, if there is one.\n        "
        cursor = self._control.textCursor()
        if before_prompt and (self._reading or not self._executing):
            self._flush_pending_stream()
            cursor._insert_mode = True
            cursor.setPosition(self._append_before_prompt_pos)
        else:
            if insert != self._insert_plain_text:
                self._flush_pending_stream()
            cursor.movePosition(QtGui.QTextCursor.End)
        result = insert(cursor, input, *args, **kwargs)
        return result

    def _append_block(self, block_format=None, before_prompt=False):
        if False:
            return 10
        ' Appends an new QTextBlock to the end of the console buffer.\n        '
        self._append_custom(self._insert_block, block_format, before_prompt)

    def _append_html(self, html, before_prompt=False):
        if False:
            i = 10
            return i + 15
        ' Appends HTML at the end of the console buffer.\n        '
        self._append_custom(self._insert_html, html, before_prompt)

    def _append_html_fetching_plain_text(self, html, before_prompt=False):
        if False:
            i = 10
            return i + 15
        ' Appends HTML, then returns the plain text version of it.\n        '
        return self._append_custom(self._insert_html_fetching_plain_text, html, before_prompt)

    def _append_plain_text(self, text, before_prompt=False):
        if False:
            print('Hello World!')
        ' Appends plain text, processing ANSI codes if enabled.\n        '
        self._append_custom(self._insert_plain_text, text, before_prompt)

    def _cancel_completion(self):
        if False:
            return 10
        ' If text completion is progress, cancel it.\n        '
        self._completion_widget.cancel_completion()

    def _clear_temporary_buffer(self):
        if False:
            i = 10
            return i + 15
        ' Clears the "temporary text" buffer, i.e. all the text following\n            the prompt region.\n        '
        cursor = self._get_prompt_cursor()
        prompt = self._continuation_prompt.lstrip()
        if self._temp_buffer_filled:
            self._temp_buffer_filled = False
            while cursor.movePosition(QtGui.QTextCursor.NextBlock):
                temp_cursor = QtGui.QTextCursor(cursor)
                temp_cursor.select(QtGui.QTextCursor.BlockUnderCursor)
                text = temp_cursor.selection().toPlainText().lstrip()
                if not text.startswith(prompt):
                    break
        else:
            return
        cursor.movePosition(QtGui.QTextCursor.Left)
        cursor.movePosition(QtGui.QTextCursor.End, QtGui.QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        if self._control.isUndoRedoEnabled():
            self._control.setUndoRedoEnabled(False)
            self._control.setUndoRedoEnabled(True)

    def _complete_with_items(self, cursor, items):
        if False:
            for i in range(10):
                print('nop')
        " Performs completion with 'items' at the specified cursor location.\n        "
        self._cancel_completion()
        if len(items) == 1:
            cursor.setPosition(self._control.textCursor().position(), QtGui.QTextCursor.KeepAnchor)
            cursor.insertText(items[0])
        elif len(items) > 1:
            current_pos = self._control.textCursor().position()
            prefix = os.path.commonprefix(items)
            if prefix:
                cursor.setPosition(current_pos, QtGui.QTextCursor.KeepAnchor)
                cursor.insertText(prefix)
                current_pos = cursor.position()
            self._completion_widget.show_items(cursor, items, prefix_length=len(prefix))

    def _fill_temporary_buffer(self, cursor, text, html=False):
        if False:
            return 10
        'fill the area below the active editting zone with text'
        current_pos = self._control.textCursor().position()
        cursor.beginEditBlock()
        self._append_plain_text('\n')
        self._page(text, html=html)
        cursor.endEditBlock()
        cursor.setPosition(current_pos)
        self._control.moveCursor(QtGui.QTextCursor.End)
        self._control.setTextCursor(cursor)
        self._temp_buffer_filled = True

    def _context_menu_make(self, pos):
        if False:
            while True:
                i = 10
        ' Creates a context menu for the given QPoint (in widget coordinates).\n        '
        menu = QtWidgets.QMenu(self)
        self.cut_action = menu.addAction('Cut', self.cut)
        self.cut_action.setEnabled(self.can_cut())
        self.cut_action.setShortcut(QtGui.QKeySequence.Cut)
        self.copy_action = menu.addAction('Copy', self.copy)
        self.copy_action.setEnabled(self.can_copy())
        self.copy_action.setShortcut(QtGui.QKeySequence.Copy)
        self.paste_action = menu.addAction('Paste', self.paste)
        self.paste_action.setEnabled(self.can_paste())
        self.paste_action.setShortcut(QtGui.QKeySequence.Paste)
        anchor = self._control.anchorAt(pos)
        if anchor:
            menu.addSeparator()
            self.copy_link_action = menu.addAction('Copy Link Address', lambda : self.copy_anchor(anchor=anchor))
            self.open_link_action = menu.addAction('Open Link', lambda : self.open_anchor(anchor=anchor))
        menu.addSeparator()
        menu.addAction(self.select_all_action)
        menu.addSeparator()
        menu.addAction(self.export_action)
        menu.addAction(self.print_action)
        return menu

    def _control_key_down(self, modifiers, include_command=False):
        if False:
            i = 10
            return i + 15
        ' Given a KeyboardModifiers flags object, return whether the Control\n        key is down.\n\n        Parameters\n        ----------\n        include_command : bool, optional (default True)\n            Whether to treat the Command key as a (mutually exclusive) synonym\n            for Control when in Mac OS.\n        '
        if sys.platform == 'darwin':
            down = include_command and modifiers & QtCore.Qt.ControlModifier
            return bool(down) ^ bool(modifiers & QtCore.Qt.MetaModifier)
        else:
            return bool(modifiers & QtCore.Qt.ControlModifier)

    def _create_control(self):
        if False:
            for i in range(10):
                print('nop')
        ' Creates and connects the underlying text widget.\n        '
        if self.custom_control:
            control = self.custom_control()
        elif self.kind == 'plain':
            control = QtWidgets.QPlainTextEdit()
        elif self.kind == 'rich':
            control = QtWidgets.QTextEdit()
            control.setAcceptRichText(False)
            control.setMouseTracking(True)
        control.setAcceptDrops(False)
        control.installEventFilter(self)
        control.viewport().installEventFilter(self)
        control.customContextMenuRequested.connect(self._custom_context_menu_requested)
        control.copyAvailable.connect(self.copy_available)
        control.redoAvailable.connect(self.redo_available)
        control.undoAvailable.connect(self.undo_available)
        layout = control.document().documentLayout()
        layout.documentSizeChanged.disconnect()
        layout.documentSizeChanged.connect(self._adjust_scrollbars)
        if self.scrollbar_visibility:
            scrollbar_policy = QtCore.Qt.ScrollBarAlwaysOn
        else:
            scrollbar_policy = QtCore.Qt.ScrollBarAlwaysOff
        control.setAttribute(QtCore.Qt.WA_InputMethodEnabled, True)
        control.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        control.setReadOnly(True)
        control.setUndoRedoEnabled(False)
        control.setVerticalScrollBarPolicy(scrollbar_policy)
        return control

    def _create_page_control(self):
        if False:
            i = 10
            return i + 15
        ' Creates and connects the underlying paging widget.\n        '
        if self.custom_page_control:
            control = self.custom_page_control()
        elif self.kind == 'plain':
            control = QtWidgets.QPlainTextEdit()
        elif self.kind == 'rich':
            control = QtWidgets.QTextEdit()
        control.installEventFilter(self)
        viewport = control.viewport()
        viewport.installEventFilter(self)
        control.setReadOnly(True)
        control.setUndoRedoEnabled(False)
        if self.scrollbar_visibility:
            scrollbar_policy = QtCore.Qt.ScrollBarAlwaysOn
        else:
            scrollbar_policy = QtCore.Qt.ScrollBarAlwaysOff
        control.setVerticalScrollBarPolicy(scrollbar_policy)
        return control

    def _event_filter_console_keypress(self, event):
        if False:
            print('Hello World!')
        ' Filter key events for the underlying text widget to create a\n            console-like interface.\n        '
        intercepted = False
        cursor = self._control.textCursor()
        position = cursor.position()
        key = event.key()
        ctrl_down = self._control_key_down(event.modifiers())
        alt_down = event.modifiers() & QtCore.Qt.AltModifier
        shift_down = event.modifiers() & QtCore.Qt.ShiftModifier
        cmd_down = sys.platform == 'darwin' and self._control_key_down(event.modifiers(), include_command=True)
        if cmd_down:
            if key == QtCore.Qt.Key_Left:
                key = QtCore.Qt.Key_Home
            elif key == QtCore.Qt.Key_Right:
                key = QtCore.Qt.Key_End
            elif key == QtCore.Qt.Key_Up:
                ctrl_down = True
                key = QtCore.Qt.Key_Home
            elif key == QtCore.Qt.Key_Down:
                ctrl_down = True
                key = QtCore.Qt.Key_End
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            intercepted = True
            self._cancel_completion()
            if self._in_buffer(position):
                if self._reading:
                    self._append_plain_text('\n')
                    self._reading = False
                    if self._reading_callback:
                        self._reading_callback()
                elif not self._executing:
                    cursor.movePosition(QtGui.QTextCursor.End, QtGui.QTextCursor.KeepAnchor)
                    at_end = len(cursor.selectedText().strip()) == 0
                    single_line = self._get_end_cursor().blockNumber() == self._get_prompt_cursor().blockNumber()
                    if (at_end or shift_down or single_line) and (not ctrl_down):
                        self.execute(interactive=not shift_down)
                    else:
                        pos = self._get_input_buffer_cursor_pos()

                        def callback(complete, indent):
                            if False:
                                i = 10
                                return i + 15
                            try:
                                cursor.beginEditBlock()
                                cursor.setPosition(position)
                                cursor.insertText('\n')
                                self._insert_continuation_prompt(cursor)
                                if indent:
                                    cursor.insertText(indent)
                            finally:
                                cursor.endEditBlock()
                            self._control.moveCursor(QtGui.QTextCursor.End)
                            self._control.setTextCursor(cursor)
                        self._register_is_complete_callback(self._get_input_buffer()[:pos], callback)
        elif ctrl_down:
            if key == QtCore.Qt.Key_G:
                self._keyboard_quit()
                intercepted = True
            elif key == QtCore.Qt.Key_K:
                if self._in_buffer(position):
                    cursor.clearSelection()
                    cursor.movePosition(QtGui.QTextCursor.EndOfLine, QtGui.QTextCursor.KeepAnchor)
                    if not cursor.hasSelection():
                        cursor.movePosition(QtGui.QTextCursor.NextBlock, QtGui.QTextCursor.KeepAnchor)
                        cursor.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, len(self._continuation_prompt))
                    self._kill_ring.kill_cursor(cursor)
                    self._set_cursor(cursor)
                intercepted = True
            elif key == QtCore.Qt.Key_L:
                self.prompt_to_top()
                intercepted = True
            elif key == QtCore.Qt.Key_O:
                if self._page_control and self._page_control.isVisible():
                    self._page_control.setFocus()
                intercepted = True
            elif key == QtCore.Qt.Key_U:
                if self._in_buffer(position):
                    cursor.clearSelection()
                    start_line = cursor.blockNumber()
                    if start_line == self._get_prompt_cursor().blockNumber():
                        offset = len(self._prompt)
                    else:
                        offset = len(self._continuation_prompt)
                    cursor.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.KeepAnchor)
                    cursor.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, offset)
                    self._kill_ring.kill_cursor(cursor)
                    self._set_cursor(cursor)
                intercepted = True
            elif key == QtCore.Qt.Key_Y:
                self._keep_cursor_in_buffer()
                self._kill_ring.yank()
                intercepted = True
            elif key in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
                if key == QtCore.Qt.Key_Backspace:
                    cursor = self._get_word_start_cursor(position)
                else:
                    cursor = self._get_word_end_cursor(position)
                cursor.setPosition(position, QtGui.QTextCursor.KeepAnchor)
                self._kill_ring.kill_cursor(cursor)
                intercepted = True
            elif key == QtCore.Qt.Key_D:
                if len(self.input_buffer) == 0 and (not self._executing):
                    self.exit_requested.emit(self)
                elif len(self._get_input_buffer(force=True)) == 0:
                    self._control.textCursor().insertText(chr(4))
                    new_event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Return, QtCore.Qt.NoModifier)
                    QtWidgets.QApplication.instance().sendEvent(self._control, new_event)
                    intercepted = True
                else:
                    new_event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Delete, QtCore.Qt.NoModifier)
                    QtWidgets.QApplication.instance().sendEvent(self._control, new_event)
                    intercepted = True
            elif key == QtCore.Qt.Key_Down:
                self._scroll_to_end()
            elif key == QtCore.Qt.Key_Up:
                self._control.verticalScrollBar().setValue(0)
        elif alt_down:
            if key == QtCore.Qt.Key_B:
                self._set_cursor(self._get_word_start_cursor(position))
                intercepted = True
            elif key == QtCore.Qt.Key_F:
                self._set_cursor(self._get_word_end_cursor(position))
                intercepted = True
            elif key == QtCore.Qt.Key_Y:
                self._kill_ring.rotate()
                intercepted = True
            elif key == QtCore.Qt.Key_Backspace:
                cursor = self._get_word_start_cursor(position)
                cursor.setPosition(position, QtGui.QTextCursor.KeepAnchor)
                self._kill_ring.kill_cursor(cursor)
                intercepted = True
            elif key == QtCore.Qt.Key_D:
                cursor = self._get_word_end_cursor(position)
                cursor.setPosition(position, QtGui.QTextCursor.KeepAnchor)
                self._kill_ring.kill_cursor(cursor)
                intercepted = True
            elif key == QtCore.Qt.Key_Delete:
                intercepted = True
            elif key == QtCore.Qt.Key_Greater:
                self._control.moveCursor(QtGui.QTextCursor.End)
                intercepted = True
            elif key == QtCore.Qt.Key_Less:
                self._control.setTextCursor(self._get_prompt_cursor())
                intercepted = True
        else:
            self._trigger_is_complete_callback()
            if shift_down:
                anchormode = QtGui.QTextCursor.KeepAnchor
            else:
                anchormode = QtGui.QTextCursor.MoveAnchor
            if key == QtCore.Qt.Key_Escape:
                self._keyboard_quit()
                intercepted = True
            elif key == QtCore.Qt.Key_Up and (not shift_down):
                if self._reading or not self._up_pressed(shift_down):
                    intercepted = True
                else:
                    prompt_line = self._get_prompt_cursor().blockNumber()
                    intercepted = cursor.blockNumber() <= prompt_line
            elif key == QtCore.Qt.Key_Down and (not shift_down):
                if self._reading or not self._down_pressed(shift_down):
                    intercepted = True
                else:
                    end_line = self._get_end_cursor().blockNumber()
                    intercepted = cursor.blockNumber() == end_line
            elif key == QtCore.Qt.Key_Tab:
                if not self._reading:
                    if self._tab_pressed():
                        self._indent(dedent=False)
                    intercepted = True
            elif key == QtCore.Qt.Key_Backtab:
                self._indent(dedent=True)
                intercepted = True
            elif key == QtCore.Qt.Key_Left and (not shift_down):
                (line, col) = (cursor.blockNumber(), cursor.columnNumber())
                if line > self._get_prompt_cursor().blockNumber() and col == len(self._continuation_prompt):
                    self._control.moveCursor(QtGui.QTextCursor.PreviousBlock, mode=anchormode)
                    self._control.moveCursor(QtGui.QTextCursor.EndOfBlock, mode=anchormode)
                    intercepted = True
                else:
                    intercepted = not self._in_buffer(position - 1)
            elif key == QtCore.Qt.Key_Right and (not shift_down):
                if position == self._get_line_end_pos():
                    cursor.movePosition(QtGui.QTextCursor.NextBlock, mode=anchormode)
                    cursor.movePosition(QtGui.QTextCursor.Right, mode=anchormode, n=len(self._continuation_prompt))
                    self._control.setTextCursor(cursor)
                else:
                    self._control.moveCursor(QtGui.QTextCursor.Right, mode=anchormode)
                intercepted = True
            elif key == QtCore.Qt.Key_Home:
                start_pos = self._get_line_start_pos()
                c = self._get_cursor()
                spaces = self._get_leading_spaces()
                if c.position() > start_pos + spaces or c.columnNumber() == len(self._continuation_prompt):
                    start_pos += spaces
                if shift_down and self._in_buffer(position):
                    if c.selectedText():
                        sel_max = max(c.selectionStart(), c.selectionEnd())
                        cursor.setPosition(sel_max, QtGui.QTextCursor.MoveAnchor)
                    cursor.setPosition(start_pos, QtGui.QTextCursor.KeepAnchor)
                else:
                    cursor.setPosition(start_pos)
                self._set_cursor(cursor)
                intercepted = True
            elif key == QtCore.Qt.Key_Backspace:
                (line, col) = (cursor.blockNumber(), cursor.columnNumber())
                if not self._reading and col == len(self._continuation_prompt) and (line > self._get_prompt_cursor().blockNumber()):
                    cursor.beginEditBlock()
                    cursor.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.KeepAnchor)
                    cursor.removeSelectedText()
                    cursor.deletePreviousChar()
                    cursor.endEditBlock()
                    intercepted = True
                else:
                    anchor = cursor.anchor()
                    if anchor == position:
                        intercepted = not self._in_buffer(position - 1)
                    else:
                        intercepted = not self._in_buffer(min(anchor, position))
            elif key == QtCore.Qt.Key_Delete:
                if not self._reading and self._in_buffer(position) and cursor.atBlockEnd() and (not cursor.hasSelection()):
                    cursor.movePosition(QtGui.QTextCursor.NextBlock, QtGui.QTextCursor.KeepAnchor)
                    cursor.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, len(self._continuation_prompt))
                    cursor.removeSelectedText()
                    intercepted = True
                else:
                    anchor = cursor.anchor()
                    intercepted = not self._in_buffer(anchor) or not self._in_buffer(position)
        if not intercepted:
            if event.matches(QtGui.QKeySequence.Copy):
                self.copy()
                intercepted = True
            elif event.matches(QtGui.QKeySequence.Cut):
                self.cut()
                intercepted = True
            elif event.matches(QtGui.QKeySequence.Paste):
                self.paste()
                intercepted = True
        if not (self._control_key_down(event.modifiers(), include_command=True) or key in (QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown) or (self._executing and (not self._reading)) or (event.text() == '' and (not (not shift_down and key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down))))):
            self._keep_cursor_in_buffer()
        return intercepted

    def _event_filter_page_keypress(self, event):
        if False:
            print('Hello World!')
        ' Filter key events for the paging widget to create console-like\n            interface.\n        '
        key = event.key()
        ctrl_down = self._control_key_down(event.modifiers())
        alt_down = event.modifiers() & QtCore.Qt.AltModifier
        if ctrl_down:
            if key == QtCore.Qt.Key_O:
                self._control.setFocus()
                return True
        elif alt_down:
            if key == QtCore.Qt.Key_Greater:
                self._page_control.moveCursor(QtGui.QTextCursor.End)
                return True
            elif key == QtCore.Qt.Key_Less:
                self._page_control.moveCursor(QtGui.QTextCursor.Start)
                return True
        elif key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            if self._splitter:
                self._page_control.hide()
                self._control.setFocus()
            else:
                self.layout().setCurrentWidget(self._control)
                self._control.document().setMaximumBlockCount(self.buffer_size)
            return True
        elif key in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return, QtCore.Qt.Key_Tab):
            new_event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_PageDown, QtCore.Qt.NoModifier)
            QtWidgets.QApplication.instance().sendEvent(self._page_control, new_event)
            return True
        elif key == QtCore.Qt.Key_Backspace:
            new_event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_PageUp, QtCore.Qt.NoModifier)
            QtWidgets.QApplication.instance().sendEvent(self._page_control, new_event)
            return True
        elif key == QtCore.Qt.Key_J:
            new_event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Down, QtCore.Qt.NoModifier)
            QtWidgets.QApplication.instance().sendEvent(self._page_control, new_event)
            return True
        elif key == QtCore.Qt.Key_K:
            new_event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Up, QtCore.Qt.NoModifier)
            QtWidgets.QApplication.instance().sendEvent(self._page_control, new_event)
            return True
        return False

    def _on_flush_pending_stream_timer(self):
        if False:
            print('Hello World!')
        ' Flush the pending stream output and change the\n        prompt position appropriately.\n        '
        cursor = self._control.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self._flush_pending_stream()
        cursor.movePosition(QtGui.QTextCursor.End)

    def _flush_pending_stream(self):
        if False:
            while True:
                i = 10
        ' Flush out pending text into the widget. '
        text = self._pending_insert_text
        self._pending_insert_text = []
        buffer_size = self._control.document().maximumBlockCount()
        if buffer_size > 0:
            text = self._get_last_lines_from_list(text, buffer_size)
        text = ''.join(text)
        t = time.time()
        self._insert_plain_text(self._get_end_cursor(), text, flush=True)
        self._pending_text_flush_interval.setInterval(int(max(100, (time.time() - t) * 1000)))

    def _get_cursor(self):
        if False:
            while True:
                i = 10
        ' Get a cursor at the current insert position.\n        '
        return self._control.textCursor()

    def _get_end_cursor(self):
        if False:
            while True:
                i = 10
        ' Get a cursor at the last character of the current cell.\n        '
        cursor = self._control.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        return cursor

    def _get_end_pos(self):
        if False:
            print('Hello World!')
        ' Get the position of the last character of the current cell.\n        '
        return self._get_end_cursor().position()

    def _get_line_start_cursor(self):
        if False:
            return 10
        ' Get a cursor at the first character of the current line.\n        '
        cursor = self._control.textCursor()
        start_line = cursor.blockNumber()
        if start_line == self._get_prompt_cursor().blockNumber():
            cursor.setPosition(self._prompt_pos)
        else:
            cursor.movePosition(QtGui.QTextCursor.StartOfLine)
            cursor.setPosition(cursor.position() + len(self._continuation_prompt))
        return cursor

    def _get_line_start_pos(self):
        if False:
            return 10
        ' Get the position of the first character of the current line.\n        '
        return self._get_line_start_cursor().position()

    def _get_line_end_cursor(self):
        if False:
            return 10
        ' Get a cursor at the last character of the current line.\n        '
        cursor = self._control.textCursor()
        cursor.movePosition(QtGui.QTextCursor.EndOfLine)
        return cursor

    def _get_line_end_pos(self):
        if False:
            while True:
                i = 10
        ' Get the position of the last character of the current line.\n        '
        return self._get_line_end_cursor().position()

    def _get_input_buffer_cursor_column(self):
        if False:
            print('Hello World!')
        ' Get the column of the cursor in the input buffer, excluding the\n            contribution by the prompt, or -1 if there is no such column.\n        '
        prompt = self._get_input_buffer_cursor_prompt()
        if prompt is None:
            return -1
        else:
            cursor = self._control.textCursor()
            return cursor.columnNumber() - len(prompt)

    def _get_input_buffer_cursor_line(self):
        if False:
            while True:
                i = 10
        ' Get the text of the line of the input buffer that contains the\n            cursor, or None if there is no such line.\n        '
        prompt = self._get_input_buffer_cursor_prompt()
        if prompt is None:
            return None
        else:
            cursor = self._control.textCursor()
            text = cursor.block().text()
            return text[len(prompt):]

    def _get_input_buffer_cursor_pos(self):
        if False:
            while True:
                i = 10
        'Get the cursor position within the input buffer.'
        cursor = self._control.textCursor()
        cursor.setPosition(self._prompt_pos, QtGui.QTextCursor.KeepAnchor)
        input_buffer = cursor.selection().toPlainText()
        return len(input_buffer.replace('\n' + self._continuation_prompt, '\n'))

    def _get_input_buffer_cursor_prompt(self):
        if False:
            i = 10
            return i + 15
        ' Returns the (plain text) prompt for line of the input buffer that\n            contains the cursor, or None if there is no such line.\n        '
        if self._executing:
            return None
        cursor = self._control.textCursor()
        if cursor.position() >= self._prompt_pos:
            if cursor.blockNumber() == self._get_prompt_cursor().blockNumber():
                return self._prompt
            else:
                return self._continuation_prompt
        else:
            return None

    def _get_last_lines(self, text, num_lines, return_count=False):
        if False:
            while True:
                i = 10
        ' Get the last specified number of lines of text (like `tail -n`).\n        If return_count is True, returns a tuple of clipped text and the\n        number of lines in the clipped text.\n        '
        pos = len(text)
        if pos < num_lines:
            if return_count:
                return (text, text.count('\n') if return_count else text)
            else:
                return text
        i = 0
        while i < num_lines:
            pos = text.rfind('\n', None, pos)
            if pos == -1:
                pos = None
                break
            i += 1
        if return_count:
            return (text[pos:], i)
        else:
            return text[pos:]

    def _get_last_lines_from_list(self, text_list, num_lines):
        if False:
            while True:
                i = 10
        ' Get the list of text clipped to last specified lines.\n        '
        ret = []
        lines_pending = num_lines
        for text in reversed(text_list):
            (text, lines_added) = self._get_last_lines(text, lines_pending, return_count=True)
            ret.append(text)
            lines_pending -= lines_added
            if lines_pending <= 0:
                break
        return ret[::-1]

    def _get_leading_spaces(self):
        if False:
            i = 10
            return i + 15
        ' Get the number of leading spaces of the current line.\n        '
        cursor = self._get_cursor()
        start_line = cursor.blockNumber()
        if start_line == self._get_prompt_cursor().blockNumber():
            offset = len(self._prompt)
        else:
            offset = len(self._continuation_prompt)
        cursor.select(QtGui.QTextCursor.LineUnderCursor)
        text = cursor.selectedText()[offset:]
        return len(text) - len(text.lstrip())

    @property
    def _prompt_pos(self):
        if False:
            return 10
        ' Find the position in the text right after the prompt.\n        '
        return min(self._prompt_cursor.position() + 1, self._get_end_pos())

    @property
    def _append_before_prompt_pos(self):
        if False:
            for i in range(10):
                print('nop')
        ' Find the position in the text right before the prompt.\n        '
        return min(self._append_before_prompt_cursor.position(), self._get_end_pos())

    def _get_prompt_cursor(self):
        if False:
            print('Hello World!')
        ' Get a cursor at the prompt position of the current cell.\n        '
        cursor = self._control.textCursor()
        cursor.setPosition(self._prompt_pos)
        return cursor

    def _get_selection_cursor(self, start, end):
        if False:
            while True:
                i = 10
        " Get a cursor with text selected between the positions 'start' and\n            'end'.\n        "
        cursor = self._control.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
        return cursor

    def _get_word_start_cursor(self, position):
        if False:
            return 10
        ' Find the start of the word to the left the given position. If a\n            sequence of non-word characters precedes the first word, skip over\n            them. (This emulates the behavior of bash, emacs, etc.)\n        '
        document = self._control.document()
        cursor = self._control.textCursor()
        line_start_pos = self._get_line_start_pos()
        if position == self._prompt_pos:
            return cursor
        elif position == line_start_pos:
            cursor = self._control.textCursor()
            cursor.setPosition(position)
            cursor.movePosition(QtGui.QTextCursor.PreviousBlock)
            cursor.movePosition(QtGui.QTextCursor.EndOfBlock)
            position = cursor.position()
            while position >= self._prompt_pos and is_whitespace(document.characterAt(position)):
                position -= 1
            cursor.setPosition(position + 1)
        else:
            position -= 1
            while position >= self._prompt_pos and position >= line_start_pos and (not is_letter_or_number(document.characterAt(position))):
                position -= 1
            while position >= self._prompt_pos and position >= line_start_pos and is_letter_or_number(document.characterAt(position)):
                position -= 1
            cursor.setPosition(position + 1)
        return cursor

    def _get_word_end_cursor(self, position):
        if False:
            print('Hello World!')
        ' Find the end of the word to the right the given position. If a\n            sequence of non-word characters precedes the first word, skip over\n            them. (This emulates the behavior of bash, emacs, etc.)\n        '
        document = self._control.document()
        cursor = self._control.textCursor()
        end_pos = self._get_end_pos()
        line_end_pos = self._get_line_end_pos()
        if position == end_pos:
            return cursor
        elif position == line_end_pos:
            cursor = self._control.textCursor()
            cursor.setPosition(position)
            cursor.movePosition(QtGui.QTextCursor.NextBlock)
            position = cursor.position() + len(self._continuation_prompt)
            while position < end_pos and is_whitespace(document.characterAt(position)):
                position += 1
            cursor.setPosition(position)
        else:
            if is_whitespace(document.characterAt(position)):
                is_indentation_whitespace = True
                back_pos = position - 1
                line_start_pos = self._get_line_start_pos()
                while back_pos >= line_start_pos:
                    if not is_whitespace(document.characterAt(back_pos)):
                        is_indentation_whitespace = False
                        break
                    back_pos -= 1
                if is_indentation_whitespace:
                    while position < end_pos and position < line_end_pos and is_whitespace(document.characterAt(position)):
                        position += 1
                    cursor.setPosition(position)
                    return cursor
            while position < end_pos and position < line_end_pos and (not is_letter_or_number(document.characterAt(position))):
                position += 1
            while position < end_pos and position < line_end_pos and is_letter_or_number(document.characterAt(position)):
                position += 1
            cursor.setPosition(position)
        return cursor

    def _indent(self, dedent=True):
        if False:
            return 10
        ' Indent/Dedent current line or current text selection.\n        '
        num_newlines = self._get_cursor().selectedText().count('\u2029')
        save_cur = self._get_cursor()
        cur = self._get_cursor()
        cur.setPosition(cur.selectionStart())
        self._control.setTextCursor(cur)
        spaces = self._get_leading_spaces()
        step = self._tab_width - spaces % self._tab_width
        cur.clearSelection()
        for _ in range(num_newlines + 1):
            self._control.setTextCursor(cur)
            cur.setPosition(self._get_line_start_pos())
            if dedent:
                spaces = min(step, self._get_leading_spaces())
                safe_step = spaces % self._tab_width
                cur.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, min(spaces, safe_step if safe_step != 0 else self._tab_width))
                cur.removeSelectedText()
            else:
                cur.insertText(' ' * step)
            cur.movePosition(QtGui.QTextCursor.Down)
        self._control.setTextCursor(save_cur)

    def _insert_continuation_prompt(self, cursor, indent=''):
        if False:
            for i in range(10):
                print('nop')
        ' Inserts new continuation prompt using the specified cursor.\n        '
        if self._continuation_prompt_html is None:
            self._insert_plain_text(cursor, self._continuation_prompt)
        else:
            self._continuation_prompt = self._insert_html_fetching_plain_text(cursor, self._continuation_prompt_html)
        if indent:
            cursor.insertText(indent)

    def _insert_block(self, cursor, block_format=None):
        if False:
            print('Hello World!')
        ' Inserts an empty QTextBlock using the specified cursor.\n        '
        if block_format is None:
            block_format = QtGui.QTextBlockFormat()
        cursor.insertBlock(block_format)

    def _insert_html(self, cursor, html):
        if False:
            for i in range(10):
                print('nop')
        ' Inserts HTML using the specified cursor in such a way that future\n            formatting is unaffected.\n        '
        cursor.beginEditBlock()
        cursor.insertHtml(html)
        cursor.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor)
        if cursor.selection().toPlainText() == ' ':
            cursor.removeSelectedText()
        else:
            cursor.movePosition(QtGui.QTextCursor.Right)
        cursor.insertText(' ', QtGui.QTextCharFormat())
        cursor.endEditBlock()

    def _insert_html_fetching_plain_text(self, cursor, html):
        if False:
            for i in range(10):
                print('nop')
        ' Inserts HTML using the specified cursor, then returns its plain text\n            version.\n        '
        cursor.beginEditBlock()
        cursor.removeSelectedText()
        start = cursor.position()
        self._insert_html(cursor, html)
        end = cursor.position()
        cursor.setPosition(start, QtGui.QTextCursor.KeepAnchor)
        text = cursor.selection().toPlainText()
        cursor.setPosition(end)
        cursor.endEditBlock()
        return text

    def _viewport_at_end(self):
        if False:
            while True:
                i = 10
        'Check if the viewport is at the end of the document.'
        viewport = self._control.viewport()
        end_scroll_pos = self._control.cursorForPosition(QtCore.QPoint(viewport.width() - 1, viewport.height() - 1)).position()
        end_doc_pos = self._get_end_pos()
        return end_doc_pos - end_scroll_pos <= 1

    def _scroll_to_end(self):
        if False:
            while True:
                i = 10
        'Scroll to the end of the document.'
        end_scroll = self._control.verticalScrollBar().maximum() - self._control.verticalScrollBar().pageStep()
        if end_scroll > self._control.verticalScrollBar().value():
            self._control.verticalScrollBar().setValue(end_scroll)

    def _insert_plain_text(self, cursor, text, flush=False):
        if False:
            for i in range(10):
                print('nop')
        ' Inserts plain text using the specified cursor, processing ANSI codes\n            if enabled.\n        '
        should_autoscroll = self._viewport_at_end()
        buffer_size = self._control.document().maximumBlockCount()
        if self._executing and (not flush) and self._pending_text_flush_interval.isActive() and (cursor.position() == self._get_end_pos()):
            self._pending_insert_text.append(text)
            if buffer_size > 0:
                self._pending_insert_text = self._get_last_lines_from_list(self._pending_insert_text, buffer_size)
            return
        if self._executing and (not self._pending_text_flush_interval.isActive()):
            self._pending_text_flush_interval.start()
        if buffer_size > 0:
            text = self._get_last_lines(text, buffer_size)
        cursor.beginEditBlock()
        if self.ansi_codes:
            for substring in self._ansi_processor.split_string(text):
                for act in self._ansi_processor.actions:
                    if act.action == 'erase':
                        remove = False
                        fill = False
                        if act.area == 'screen':
                            cursor.select(QtGui.QTextCursor.Document)
                            remove = True
                        if act.area == 'line':
                            if act.erase_to == 'all':
                                cursor.select(QtGui.QTextCursor.LineUnderCursor)
                                remove = True
                            elif act.erase_to == 'start':
                                cursor.movePosition(QtGui.QTextCursor.StartOfLine, QtGui.QTextCursor.KeepAnchor)
                                remove = True
                                fill = True
                            elif act.erase_to == 'end':
                                cursor.movePosition(QtGui.QTextCursor.EndOfLine, QtGui.QTextCursor.KeepAnchor)
                                remove = True
                        if remove:
                            nspace = cursor.selectionEnd() - cursor.selectionStart() if fill else 0
                            cursor.removeSelectedText()
                            if nspace > 0:
                                cursor.insertText(' ' * nspace)
                    elif act.action == 'scroll' and act.unit == 'page':
                        cursor.insertText('\n')
                        cursor.endEditBlock()
                        self._set_top_cursor(cursor)
                        cursor.joinPreviousEditBlock()
                        cursor.deletePreviousChar()
                        if os.name == 'nt':
                            cursor.select(QtGui.QTextCursor.Document)
                            cursor.removeSelectedText()
                    elif act.action == 'carriage-return':
                        cursor.movePosition(QtGui.QTextCursor.StartOfLine, QtGui.QTextCursor.MoveAnchor)
                    elif act.action == 'beep':
                        QtWidgets.QApplication.instance().beep()
                    elif act.action == 'backspace':
                        if not cursor.atBlockStart():
                            cursor.movePosition(QtGui.QTextCursor.PreviousCharacter, QtGui.QTextCursor.MoveAnchor)
                    elif act.action == 'newline':
                        cursor.movePosition(QtGui.QTextCursor.EndOfLine)
                if substring is not None:
                    format = self._ansi_processor.get_format()
                    if not (hasattr(cursor, '_insert_mode') and cursor._insert_mode):
                        pos = cursor.position()
                        cursor2 = QtGui.QTextCursor(cursor)
                        cursor2.movePosition(QtGui.QTextCursor.EndOfLine)
                        remain = cursor2.position() - pos
                        n = len(substring)
                        swallow = min(n, remain)
                        cursor.setPosition(pos + swallow, QtGui.QTextCursor.KeepAnchor)
                    cursor.insertText(substring, format)
        else:
            cursor.insertText(text)
        cursor.endEditBlock()
        if should_autoscroll:
            self._scroll_to_end()

    def _insert_plain_text_into_buffer(self, cursor, text):
        if False:
            print('Hello World!')
        ' Inserts text into the input buffer using the specified cursor (which\n            must be in the input buffer), ensuring that continuation prompts are\n            inserted as necessary.\n        '
        lines = text.splitlines(True)
        if lines:
            if lines[-1].endswith('\n'):
                lines.append('')
            cursor.beginEditBlock()
            cursor.insertText(lines[0])
            for line in lines[1:]:
                if self._continuation_prompt_html is None:
                    cursor.insertText(self._continuation_prompt)
                else:
                    self._continuation_prompt = self._insert_html_fetching_plain_text(cursor, self._continuation_prompt_html)
                cursor.insertText(line)
            cursor.endEditBlock()

    def _in_buffer(self, position):
        if False:
            while True:
                i = 10
        '\n        Returns whether the specified position is inside the editing region.\n        '
        return position == self._move_position_in_buffer(position)

    def _move_position_in_buffer(self, position):
        if False:
            i = 10
            return i + 15
        '\n        Return the next position in buffer.\n        '
        cursor = self._control.textCursor()
        cursor.setPosition(position)
        line = cursor.blockNumber()
        prompt_line = self._get_prompt_cursor().blockNumber()
        if line == prompt_line:
            if position >= self._prompt_pos:
                return position
            return self._prompt_pos
        if line > prompt_line:
            cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
            prompt_pos = cursor.position() + len(self._continuation_prompt)
            if position >= prompt_pos:
                return position
            return prompt_pos
        return self._prompt_pos

    def _keep_cursor_in_buffer(self):
        if False:
            return 10
        ' Ensures that the cursor is inside the editing region. Returns\n            whether the cursor was moved.\n        '
        cursor = self._control.textCursor()
        endpos = cursor.selectionEnd()
        if endpos < self._prompt_pos:
            cursor.setPosition(endpos)
            line = cursor.blockNumber()
            prompt_line = self._get_prompt_cursor().blockNumber()
            if line == prompt_line:
                cursor.setPosition(self._prompt_pos)
            else:
                cursor.movePosition(QtGui.QTextCursor.End)
            self._control.setTextCursor(cursor)
            return True
        startpos = cursor.selectionStart()
        new_endpos = self._move_position_in_buffer(endpos)
        new_startpos = self._move_position_in_buffer(startpos)
        if new_endpos == endpos and new_startpos == startpos:
            return False
        cursor.setPosition(new_startpos)
        cursor.setPosition(new_endpos, QtGui.QTextCursor.KeepAnchor)
        self._control.setTextCursor(cursor)
        return True

    def _keyboard_quit(self):
        if False:
            for i in range(10):
                print('nop')
        ' Cancels the current editing task ala Ctrl-G in Emacs.\n        '
        if self._temp_buffer_filled:
            self._cancel_completion()
            self._clear_temporary_buffer()
        else:
            self.input_buffer = ''

    def _page(self, text, html=False):
        if False:
            i = 10
            return i + 15
        ' Displays text using the pager if it exceeds the height of the\n        viewport.\n\n        Parameters\n        ----------\n        html : bool, optional (default False)\n            If set, the text will be interpreted as HTML instead of plain text.\n        '
        line_height = QtGui.QFontMetrics(self.font).height()
        minlines = self._control.viewport().height() / line_height
        if self.paging != 'none' and re.match('(?:[^\n]*\n){%i}' % minlines, text):
            if self.paging == 'custom':
                self.custom_page_requested.emit(text)
            else:
                self._control.document().setMaximumBlockCount(0)
                self._page_control.clear()
                cursor = self._page_control.textCursor()
                if html:
                    self._insert_html(cursor, text)
                else:
                    self._insert_plain_text(cursor, text)
                self._page_control.moveCursor(QtGui.QTextCursor.Start)
                self._page_control.viewport().resize(self._control.size())
                if self._splitter:
                    self._page_control.show()
                    self._page_control.setFocus()
                else:
                    self.layout().setCurrentWidget(self._page_control)
        elif html:
            self._append_html(text)
        else:
            self._append_plain_text(text)

    def _set_paging(self, paging):
        if False:
            print('Hello World!')
        '\n        Change the pager to `paging` style.\n\n        Parameters\n        ----------\n        paging : string\n            Either "hsplit", "vsplit", or "inside"\n        '
        if self._splitter is None:
            raise NotImplementedError('can only switch if --paging=hsplit or\n                    --paging=vsplit is used.')
        if paging == 'hsplit':
            self._splitter.setOrientation(QtCore.Qt.Horizontal)
        elif paging == 'vsplit':
            self._splitter.setOrientation(QtCore.Qt.Vertical)
        elif paging == 'inside':
            raise NotImplementedError("switching to 'inside' paging not\n                    supported yet.")
        else:
            raise ValueError("unknown paging method '%s'" % paging)
        self.paging = paging

    def _prompt_finished(self):
        if False:
            while True:
                i = 10
        ' Called immediately after a prompt is finished, i.e. when some input\n            will be processed and a new prompt displayed.\n        '
        self._control.setReadOnly(True)
        self._prompt_finished_hook()

    def _prompt_started(self):
        if False:
            for i in range(10):
                print('nop')
        ' Called immediately after a new prompt is displayed.\n        '
        self._control.document().setMaximumBlockCount(0)
        self._control.setUndoRedoEnabled(True)
        self._control.setReadOnly(False)
        self._control.setAttribute(QtCore.Qt.WA_InputMethodEnabled, True)
        if not self._reading:
            self._executing = False
        self._prompt_started_hook()
        if self._input_buffer_pending:
            self.input_buffer = self._input_buffer_pending
            self._input_buffer_pending = ''
        self._control.moveCursor(QtGui.QTextCursor.End)

    def _readline(self, prompt='', callback=None, password=False):
        if False:
            for i in range(10):
                print('nop')
        ' Reads one line of input from the user.\n\n        Parameters\n        ----------\n        prompt : str, optional\n            The prompt to print before reading the line.\n\n        callback : callable, optional\n            A callback to execute with the read line. If not specified, input is\n            read *synchronously* and this method does not return until it has\n            been read.\n\n        Returns\n        -------\n        If a callback is specified, returns nothing. Otherwise, returns the\n        input string with the trailing newline stripped.\n        '
        if self._reading:
            raise RuntimeError('Cannot read a line. Widget is already reading.')
        if not callback and (not self.isVisible()):
            raise RuntimeError('Cannot synchronously read a line if the widget is not visible!')
        self._reading = True
        if password:
            self._show_prompt('Warning: QtConsole does not support password mode, the text you type will be visible.', newline=True)
        if 'ipdb' not in prompt.lower():
            self._show_prompt(prompt, newline=False, separator=False)
        else:
            self._show_prompt(prompt, newline=False)
        if callback is None:
            self._reading_callback = None
            while self._reading:
                QtCore.QCoreApplication.processEvents()
            return self._get_input_buffer(force=True).rstrip('\n')
        else:
            self._reading_callback = lambda : callback(self._get_input_buffer(force=True).rstrip('\n'))

    def _set_continuation_prompt(self, prompt, html=False):
        if False:
            i = 10
            return i + 15
        ' Sets the continuation prompt.\n\n        Parameters\n        ----------\n        prompt : str\n            The prompt to show when more input is needed.\n\n        html : bool, optional (default False)\n            If set, the prompt will be inserted as formatted HTML. Otherwise,\n            the prompt will be treated as plain text, though ANSI color codes\n            will be handled.\n        '
        if html:
            self._continuation_prompt_html = prompt
        else:
            self._continuation_prompt = prompt
            self._continuation_prompt_html = None

    def _set_cursor(self, cursor):
        if False:
            print('Hello World!')
        ' Convenience method to set the current cursor.\n        '
        self._control.setTextCursor(cursor)

    def _set_top_cursor(self, cursor):
        if False:
            return 10
        ' Scrolls the viewport so that the specified cursor is at the top.\n        '
        scrollbar = self._control.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        original_cursor = self._control.textCursor()
        self._control.setTextCursor(cursor)
        self._control.ensureCursorVisible()
        self._control.setTextCursor(original_cursor)

    def _show_prompt(self, prompt=None, html=False, newline=True, separator=True):
        if False:
            print('Hello World!')
        ' Writes a new prompt at the end of the buffer.\n\n        Parameters\n        ----------\n        prompt : str, optional\n            The prompt to show. If not specified, the previous prompt is used.\n\n        html : bool, optional (default False)\n            Only relevant when a prompt is specified. If set, the prompt will\n            be inserted as formatted HTML. Otherwise, the prompt will be treated\n            as plain text, though ANSI color codes will be handled.\n\n        newline : bool, optional (default True)\n            If set, a new line will be written before showing the prompt if\n            there is not already a newline at the end of the buffer.\n\n        separator : bool, optional (default True)\n            If set, a separator will be written before the prompt.\n        '
        self._flush_pending_stream()
        if sys.platform == 'darwin':
            if not os.environ.get('QTCONSOLE_TESTING'):
                QtCore.QCoreApplication.processEvents()
        else:
            QtCore.QCoreApplication.processEvents()
        cursor = self._get_end_cursor()
        if cursor.position() == 0:
            move_forward = False
        else:
            move_forward = True
            self._append_before_prompt_cursor.setPosition(cursor.position() - 1)
        if newline and cursor.position() > 0:
            cursor.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor)
            if cursor.selection().toPlainText() != '\n':
                self._append_block()
        if separator:
            self._append_plain_text(self._prompt_sep)
        if prompt is None:
            if self._prompt_html is None:
                self._append_plain_text(self._prompt)
            else:
                self._append_html(self._prompt_html)
        elif html:
            self._prompt = self._append_html_fetching_plain_text(prompt)
            self._prompt_html = prompt
        else:
            self._append_plain_text(prompt)
            self._prompt = prompt
            self._prompt_html = None
        self._flush_pending_stream()
        self._prompt_cursor.setPosition(self._get_end_pos() - 1)
        if move_forward:
            self._append_before_prompt_cursor.setPosition(self._append_before_prompt_cursor.position() + 1)
        self._prompt_started()

    def _adjust_scrollbars(self):
        if False:
            for i in range(10):
                print('nop')
        ' Expands the vertical scrollbar beyond the range set by Qt.\n        '
        document = self._control.document()
        scrollbar = self._control.verticalScrollBar()
        viewport_height = self._control.viewport().height()
        if isinstance(self._control, QtWidgets.QPlainTextEdit):
            maximum = max(0, document.lineCount() - 1)
            step = viewport_height / self._control.fontMetrics().lineSpacing()
        else:
            maximum = document.size().height()
            step = viewport_height
        diff = maximum - scrollbar.maximum()
        scrollbar.setRange(0, round(maximum))
        scrollbar.setPageStep(round(step))
        if diff < 0 and document.blockCount() == document.maximumBlockCount():
            scrollbar.setValue(round(scrollbar.value() + diff))

    def _custom_context_menu_requested(self, pos):
        if False:
            i = 10
            return i + 15
        ' Shows a context menu at the given QPoint (in widget coordinates).\n        '
        menu = self._context_menu_make(pos)
        menu.exec_(self._control.mapToGlobal(pos))