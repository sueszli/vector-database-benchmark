"""A FrontendWidget that emulates a repl for a Jupyter kernel.

This supports the additional functionality provided by Jupyter kernel.
"""
from collections import namedtuple
from subprocess import Popen
import sys
import time
from warnings import warn
from qtpy import QtCore, QtGui
from IPython.lib.lexers import IPythonLexer, IPython3Lexer
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from qtconsole import __version__
from traitlets import Bool, Unicode, observe, default
from .frontend_widget import FrontendWidget
from . import styles
default_in_prompt = 'In [<span class="in-prompt-number">%i</span>]: '
default_out_prompt = 'Out[<span class="out-prompt-number">%i</span>]: '
default_input_sep = '\n'
default_output_sep = ''
default_output_sep2 = ''
zmq_shell_source = 'ipykernel.zmqshell.ZMQInteractiveShell'
if sys.platform.startswith('win'):
    default_editor = 'notepad'
else:
    default_editor = ''

class IPythonWidget(FrontendWidget):
    """Dummy class for config inheritance. Destroyed below."""

class JupyterWidget(IPythonWidget):
    """A FrontendWidget for a Jupyter kernel."""
    custom_edit = Bool(False)
    custom_edit_requested = QtCore.Signal(object, object)
    editor = Unicode(default_editor, config=True, help='\n        A command for invoking a GUI text editor. If the string contains a\n        {filename} format specifier, it will be used. Otherwise, the filename\n        will be appended to the end the command. To use a terminal text editor,\n        the command should launch a new terminal, e.g.\n        ``"gnome-terminal -- vim"``.\n        ')
    editor_line = Unicode(config=True, help='\n        The editor command to use when a specific line number is requested. The\n        string should contain two format specifiers: {line} and {filename}. If\n        this parameter is not specified, the line number option to the %edit\n        magic will be ignored.\n        ')
    style_sheet = Unicode(config=True, help='\n        A CSS stylesheet. The stylesheet can contain classes for:\n            1. Qt: QPlainTextEdit, QFrame, QWidget, etc\n            2. Pygments: .c, .k, .o, etc. (see PygmentsHighlighter)\n            3. QtConsole: .error, .in-prompt, .out-prompt, etc\n        ')
    syntax_style = Unicode(config=True, help='\n        If not empty, use this Pygments style for syntax highlighting.\n        Otherwise, the style sheet is queried for Pygments style\n        information.\n        ')
    in_prompt = Unicode(default_in_prompt, config=True)
    out_prompt = Unicode(default_out_prompt, config=True)
    input_sep = Unicode(default_input_sep, config=True)
    output_sep = Unicode(default_output_sep, config=True)
    output_sep2 = Unicode(default_output_sep2, config=True)
    _PromptBlock = namedtuple('_PromptBlock', ['block', 'length', 'number'])
    _payload_source_edit = 'edit_magic'
    _payload_source_exit = 'ask_exit'
    _payload_source_next_input = 'set_next_input'
    _payload_source_page = 'page'
    _retrying_history_request = False
    _starting = False

    def __init__(self, *args, **kw):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kw)
        self._payload_handlers = {self._payload_source_edit: self._handle_payload_edit, self._payload_source_exit: self._handle_payload_exit, self._payload_source_page: self._handle_payload_page, self._payload_source_next_input: self._handle_payload_next_input}
        self._previous_prompt_obj = None
        self._keep_kernel_on_exit = None
        if self.style_sheet:
            self._style_sheet_changed()
            self._syntax_style_changed()
        else:
            self.set_default_style()
        self.language_name = None
        self._prompt_requested = False

    def _handle_complete_reply(self, rep):
        if False:
            return 10
        "Support Jupyter's improved completion machinery.\n        "
        self.log.debug('complete: %s', rep.get('content', ''))
        cursor = self._get_cursor()
        info = self._request_info.get('complete')
        if info and info.id == rep['parent_header']['msg_id'] and (info.pos == self._get_input_buffer_cursor_pos()) and (info.code == self.input_buffer):
            content = rep['content']
            matches = content['matches']
            start = content['cursor_start']
            end = content['cursor_end']
            start = max(start, 0)
            end = max(end, start)
            cursor_pos = self._get_input_buffer_cursor_pos()
            if end < cursor_pos:
                cursor.movePosition(QtGui.QTextCursor.Left, n=cursor_pos - end)
            elif end > cursor_pos:
                cursor.movePosition(QtGui.QTextCursor.Right, n=end - cursor_pos)
            self._control.setTextCursor(cursor)
            offset = end - start
            cursor.movePosition(QtGui.QTextCursor.Left, n=offset)
            self._complete_with_items(cursor, matches)

    def _handle_execute_reply(self, msg):
        if False:
            for i in range(10):
                print('nop')
        'Support prompt requests.\n        '
        msg_id = msg['parent_header'].get('msg_id')
        info = self._request_info['execute'].get(msg_id)
        if info and info.kind == 'prompt':
            self._prompt_requested = False
            content = msg['content']
            if content['status'] == 'aborted':
                self._show_interpreter_prompt()
            else:
                number = content['execution_count'] + 1
                self._show_interpreter_prompt(number)
            self._request_info['execute'].pop(msg_id)
        else:
            super()._handle_execute_reply(msg)

    def _handle_history_reply(self, msg):
        if False:
            for i in range(10):
                print('nop')
        ' Handle history tail replies, which are only supported\n            by Jupyter kernels.\n        '
        content = msg['content']
        if 'history' not in content:
            self.log.error('History request failed: %r' % content)
            if content.get('status', '') == 'aborted' and (not self._retrying_history_request):
                self.log.error('Retrying aborted history request')
                self._retrying_history_request = True
                time.sleep(0.25)
                self.kernel_client.history(hist_access_type='tail', n=1000)
            else:
                self._retrying_history_request = False
            return
        self._retrying_history_request = False
        history_items = content['history']
        self.log.debug('Received history reply with %i entries', len(history_items))
        items = []
        last_cell = ''
        for (_, _, cell) in history_items:
            cell = cell.rstrip()
            if cell != last_cell:
                items.append(cell)
                last_cell = cell
        self._set_history(items)

    def _insert_other_input(self, cursor, content, remote=True):
        if False:
            return 10
        'Insert function for input from other frontends'
        n = content.get('execution_count', 0)
        prompt = self._make_in_prompt(n, remote=remote)
        cont_prompt = self._make_continuation_prompt(self._prompt, remote=remote)
        cursor.insertText('\n')
        for (i, line) in enumerate(content['code'].strip().split('\n')):
            if i == 0:
                self._insert_html(cursor, prompt)
            else:
                self._insert_html(cursor, cont_prompt)
            self._insert_plain_text(cursor, line + '\n')
        self._update_prompt(n + 1)

    def _handle_execute_input(self, msg):
        if False:
            print('Hello World!')
        'Handle an execute_input message'
        self.log.debug('execute_input: %s', msg.get('content', ''))
        if self.include_output(msg):
            self._append_custom(self._insert_other_input, msg['content'], before_prompt=True)
        elif not self._prompt:
            self._append_custom(self._insert_other_input, msg['content'], before_prompt=True, remote=False)

    def _handle_execute_result(self, msg):
        if False:
            for i in range(10):
                print('nop')
        'Handle an execute_result message'
        self.log.debug('execute_result: %s', msg.get('content', ''))
        if self.include_output(msg):
            self.flush_clearoutput()
            content = msg['content']
            prompt_number = content.get('execution_count', 0)
            data = content['data']
            if 'text/plain' in data:
                self._append_plain_text(self.output_sep, before_prompt=True)
                self._append_html(self._make_out_prompt(prompt_number, remote=not self.from_here(msg)), before_prompt=True)
                text = data['text/plain']
                if '\n' in text and (not self.output_sep.endswith('\n')):
                    self._append_plain_text('\n', before_prompt=True)
                self._append_plain_text(text + self.output_sep2, before_prompt=True)
                if not self.from_here(msg):
                    self._append_plain_text('\n', before_prompt=True)

    def _handle_display_data(self, msg):
        if False:
            return 10
        'The base handler for the ``display_data`` message.'
        if self.include_output(msg):
            self.flush_clearoutput()
            data = msg['content']['data']
            if 'text/plain' in data:
                text = data['text/plain']
                self._append_plain_text(text, True)
            self._append_plain_text('\n', True)

    def _handle_kernel_info_reply(self, rep):
        if False:
            while True:
                i = 10
        'Handle kernel info replies.'
        content = rep['content']
        self.language_name = content['language_info']['name']
        pygments_lexer = content['language_info'].get('pygments_lexer', '')
        try:
            if pygments_lexer == 'ipython3':
                lexer = IPython3Lexer()
            elif pygments_lexer == 'ipython2':
                lexer = IPythonLexer()
            else:
                lexer = get_lexer_by_name(self.language_name)
            self._highlighter._lexer = lexer
        except ClassNotFound:
            pass
        self.kernel_banner = content.get('banner', '')
        if self._starting:
            self._starting = False
            super()._started_channels()

    def _started_channels(self):
        if False:
            print('Hello World!')
        'Make a history request'
        self._starting = True
        self.kernel_client.kernel_info()
        self.kernel_client.history(hist_access_type='tail', n=1000)

    def _process_execute_error(self, msg):
        if False:
            i = 10
            return i + 15
        'Handle an execute_error message'
        self.log.debug('execute_error: %s', msg.get('content', ''))
        content = msg['content']
        traceback = '\n'.join(content['traceback']) + '\n'
        if False:
            traceback = traceback.replace(' ', '&nbsp;')
            traceback = traceback.replace('\n', '<br/>')
            ename = content['ename']
            ename_styled = '<span class="error">%s</span>' % ename
            traceback = traceback.replace(ename, ename_styled)
            self._append_html(traceback)
        else:
            self._append_plain_text(traceback, before_prompt=not self.from_here(msg))

    def _process_execute_payload(self, item):
        if False:
            for i in range(10):
                print('nop')
        ' Reimplemented to dispatch payloads to handler methods.\n        '
        handler = self._payload_handlers.get(item['source'])
        if handler is None:
            return False
        else:
            handler(item)
            return True

    def _show_interpreter_prompt(self, number=None):
        if False:
            return 10
        ' Reimplemented for IPython-style prompts.\n        '
        if number is None:
            if self._prompt_requested:
                return
            self._prompt_requested = True
            msg_id = self.kernel_client.execute('', silent=True)
            info = self._ExecutionRequest(msg_id, 'prompt', False)
            self._request_info['execute'][msg_id] = info
            return
        self._prompt_sep = self.input_sep
        self._show_prompt(self._make_in_prompt(number), html=True)
        block = self._control.document().lastBlock()
        length = len(self._prompt)
        self._previous_prompt_obj = self._PromptBlock(block, length, number)
        self._set_continuation_prompt(self._make_continuation_prompt(self._prompt), html=True)

    def _update_prompt(self, new_prompt_number):
        if False:
            print('Hello World!')
        'Replace the last displayed prompt with a new one.'
        if self._previous_prompt_obj is None:
            return
        block = self._previous_prompt_obj.block
        if block.isValid() and block.text():
            cursor = QtGui.QTextCursor(block)
            cursor.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, self._previous_prompt_obj.length)
            prompt = self._make_in_prompt(new_prompt_number)
            self._prompt = self._insert_html_fetching_plain_text(cursor, prompt)
            self._highlighter.rehighlightBlock(cursor.block())
            self._prompt_cursor.setPosition(cursor.position() - 1)
            block = self._control.document().lastBlock()
            length = len(self._prompt)
            self._previous_prompt_obj = self._PromptBlock(block, length, new_prompt_number)

    def _show_interpreter_prompt_for_reply(self, msg):
        if False:
            return 10
        ' Reimplemented for IPython-style prompts.\n        '
        content = msg['content']
        if content['status'] == 'aborted':
            if self._previous_prompt_obj:
                previous_prompt_number = self._previous_prompt_obj.number
            else:
                previous_prompt_number = 0
        else:
            previous_prompt_number = content['execution_count']
        if self._previous_prompt_obj and self._previous_prompt_obj.number != previous_prompt_number:
            self._update_prompt(previous_prompt_number)
            self._previous_prompt_obj = None
        self._show_interpreter_prompt(previous_prompt_number + 1)

    def set_default_style(self, colors='lightbg'):
        if False:
            while True:
                i = 10
        ' Sets the widget style to the class defaults.\n\n        Parameters\n        ----------\n        colors : str, optional (default lightbg)\n            Whether to use the default light background or dark\n            background or B&W style.\n        '
        colors = colors.lower()
        if colors == 'lightbg':
            self.style_sheet = styles.default_light_style_sheet
            self.syntax_style = styles.default_light_syntax_style
        elif colors == 'linux':
            self.style_sheet = styles.default_dark_style_sheet
            self.syntax_style = styles.default_dark_syntax_style
        elif colors == 'nocolor':
            self.style_sheet = styles.default_bw_style_sheet
            self.syntax_style = styles.default_bw_syntax_style
        else:
            raise KeyError('No such color scheme: %s' % colors)

    def _edit(self, filename, line=None):
        if False:
            while True:
                i = 10
        ' Opens a Python script for editing.\n\n        Parameters\n        ----------\n        filename : str\n            A path to a local system file.\n\n        line : int, optional\n            A line of interest in the file.\n        '
        if self.custom_edit:
            self.custom_edit_requested.emit(filename, line)
        elif not self.editor:
            self._append_plain_text('No default editor available.\nSpecify a GUI text editor in the `JupyterWidget.editor` configurable to enable the %edit magic')
        else:
            try:
                filename = '"%s"' % filename
                if line and self.editor_line:
                    command = self.editor_line.format(filename=filename, line=line)
                else:
                    try:
                        command = self.editor.format()
                    except KeyError:
                        command = self.editor.format(filename=filename)
                    else:
                        command += ' ' + filename
            except KeyError:
                self._append_plain_text('Invalid editor command.\n')
            else:
                try:
                    Popen(command, shell=True)
                except OSError:
                    msg = 'Opening editor with command "%s" failed.\n'
                    self._append_plain_text(msg % command)

    def _make_in_prompt(self, number, remote=False):
        if False:
            for i in range(10):
                print('nop')
        ' Given a prompt number, returns an HTML In prompt.\n        '
        try:
            body = self.in_prompt % number
        except TypeError:
            from xml.sax.saxutils import escape
            body = escape(self.in_prompt)
        if remote:
            body = self.other_output_prefix + body
        return '<span class="in-prompt">%s</span>' % body

    def _make_continuation_prompt(self, prompt, remote=False):
        if False:
            i = 10
            return i + 15
        ' Given a plain text version of an In prompt, returns an HTML\n            continuation prompt.\n        '
        end_chars = '...: '
        space_count = len(prompt.lstrip('\n')) - len(end_chars)
        if remote:
            space_count += len(self.other_output_prefix.rsplit('\n')[-1])
        body = '&nbsp;' * space_count + end_chars
        return '<span class="in-prompt">%s</span>' % body

    def _make_out_prompt(self, number, remote=False):
        if False:
            for i in range(10):
                print('nop')
        ' Given a prompt number, returns an HTML Out prompt.\n        '
        try:
            body = self.out_prompt % number
        except TypeError:
            from xml.sax.saxutils import escape
            body = escape(self.out_prompt)
        if remote:
            body = self.other_output_prefix + body
        return '<span class="out-prompt">%s</span>' % body

    def _handle_payload_edit(self, item):
        if False:
            for i in range(10):
                print('nop')
        self._edit(item['filename'], item['line_number'])

    def _handle_payload_exit(self, item):
        if False:
            return 10
        self._keep_kernel_on_exit = item['keepkernel']
        self.exit_requested.emit(self)

    def _handle_payload_next_input(self, item):
        if False:
            while True:
                i = 10
        self.input_buffer = item['text']

    def _handle_payload_page(self, item):
        if False:
            while True:
                i = 10
        data = item['data']
        if 'text/html' in data and self.kind == 'rich':
            self._page(data['text/html'], html=True)
        else:
            self._page(data['text/plain'], html=False)

    @observe('style_sheet')
    def _style_sheet_changed(self, changed=None):
        if False:
            i = 10
            return i + 15
        ' Set the style sheets of the underlying widgets.\n        '
        self.setStyleSheet(self.style_sheet)
        if self._control is not None:
            self._control.document().setDefaultStyleSheet(self.style_sheet)
        if self._page_control is not None:
            self._page_control.document().setDefaultStyleSheet(self.style_sheet)

    @observe('syntax_style')
    def _syntax_style_changed(self, changed=None):
        if False:
            for i in range(10):
                print('nop')
        ' Set the style for the syntax highlighter.\n        '
        if self._highlighter is None:
            return
        if self.syntax_style:
            self._highlighter.set_style(self.syntax_style)
            self._ansi_processor.set_background_color(self.syntax_style)
        else:
            self._highlighter.set_style_sheet(self.style_sheet)

    @default('banner')
    def _banner_default(self):
        if False:
            i = 10
            return i + 15
        return 'Jupyter QtConsole {version}\n'.format(version=__version__)

class IPythonWidget(JupyterWidget):
    """Deprecated class; use JupyterWidget."""

    def __init__(self, *a, **kw):
        if False:
            while True:
                i = 10
        warn('IPythonWidget is deprecated; use JupyterWidget', DeprecationWarning)
        super().__init__(*a, **kw)