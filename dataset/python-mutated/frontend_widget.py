"""Frontend widget for the Qt Console"""
from collections import namedtuple
import sys
import uuid
import re
from qtpy import QtCore, QtGui, QtWidgets
from qtconsole.base_frontend_mixin import BaseFrontendMixin
from traitlets import Any, Bool, Instance, Unicode, DottedObjectName, default
from .bracket_matcher import BracketMatcher
from .call_tip_widget import CallTipWidget
from .history_console_widget import HistoryConsoleWidget
from .pygments_highlighter import PygmentsHighlighter
from .util import import_item

class FrontendHighlighter(PygmentsHighlighter):
    """ A PygmentsHighlighter that understands and ignores prompts.
    """

    def __init__(self, frontend, lexer=None):
        if False:
            while True:
                i = 10
        super().__init__(frontend._control.document(), lexer=lexer)
        self._current_offset = 0
        self._frontend = frontend
        self.highlighting_on = False
        self._classic_prompt_re = re.compile('^(%s)?([ \\t]*>>> |^[ \\t]*\\.\\.\\. )' % re.escape(frontend.other_output_prefix))
        self._ipy_prompt_re = re.compile('^(%s)?([ \\t]*In \\[\\d+\\]: |[ \\t]*\\ \\ \\ \\.\\.\\.+: )' % re.escape(frontend.other_output_prefix))

    def transform_classic_prompt(self, line):
        if False:
            return 10
        "Handle inputs that start with '>>> ' syntax."
        if not line or line.isspace():
            return line
        m = self._classic_prompt_re.match(line)
        if m:
            return line[len(m.group(0)):]
        else:
            return line

    def transform_ipy_prompt(self, line):
        if False:
            return 10
        'Handle inputs that start classic IPython prompt syntax.'
        if not line or line.isspace():
            return line
        m = self._ipy_prompt_re.match(line)
        if m:
            return line[len(m.group(0)):]
        else:
            return line

    def highlightBlock(self, string):
        if False:
            i = 10
            return i + 15
        ' Highlight a block of text. Reimplemented to highlight selectively.\n        '
        if not hasattr(self, 'highlighting_on') or not self.highlighting_on:
            return
        current_block = self.currentBlock()
        string = current_block.text()
        string = string.replace('\xa0', ' ')
        without_prompt = self.transform_ipy_prompt(string)
        diff = len(string) - len(without_prompt)
        if diff > 0:
            self._current_offset = diff
            super().highlightBlock(without_prompt)

    def rehighlightBlock(self, block):
        if False:
            print('Hello World!')
        ' Reimplemented to temporarily enable highlighting if disabled.\n        '
        old = self.highlighting_on
        self.highlighting_on = True
        super().rehighlightBlock(block)
        self.highlighting_on = old

    def setFormat(self, start, count, format):
        if False:
            print('Hello World!')
        ' Reimplemented to highlight selectively.\n        '
        start += self._current_offset
        super().setFormat(start, count, format)

class FrontendWidget(HistoryConsoleWidget, BaseFrontendMixin):
    """ A Qt frontend for a generic Python kernel.
    """
    banner = Unicode(config=True)
    kernel_banner = Unicode()
    _display_banner = Bool(False)
    custom_interrupt = Bool(False)
    custom_interrupt_requested = QtCore.Signal()
    custom_restart = Bool(False)
    custom_restart_kernel_died = QtCore.Signal(float)
    custom_restart_requested = QtCore.Signal()
    enable_calltips = Bool(True, config=True, help='Whether to draw information calltips on open-parentheses.')
    clear_on_kernel_restart = Bool(True, config=True, help='Whether to clear the console when the kernel is restarted')
    confirm_restart = Bool(True, config=True, help='Whether to ask for user confirmation when restarting kernel')
    lexer_class = DottedObjectName(config=True, help='The pygments lexer class to use.')

    def _lexer_class_changed(self, name, old, new):
        if False:
            print('Hello World!')
        lexer_class = import_item(new)
        self.lexer = lexer_class()

    def _lexer_class_default(self):
        if False:
            print('Hello World!')
        return 'pygments.lexers.Python3Lexer'
    lexer = Any()

    def _lexer_default(self):
        if False:
            print('Hello World!')
        lexer_class = import_item(self.lexer_class)
        return lexer_class()
    executing = QtCore.Signal(object)
    executed = QtCore.Signal(object)
    exit_requested = QtCore.Signal(object)
    _CallTipRequest = namedtuple('_CallTipRequest', ['id', 'pos'])
    _CompletionRequest = namedtuple('_CompletionRequest', ['id', 'code', 'pos'])
    _ExecutionRequest = namedtuple('_ExecutionRequest', ['id', 'kind', 'hidden'])
    _local_kernel = False
    _highlighter = Instance(FrontendHighlighter, allow_none=True)

    def __init__(self, local_kernel=_local_kernel, *args, **kw):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kw)
        self._bracket_matcher = BracketMatcher(self._control)
        self._call_tip_widget = CallTipWidget(self._control)
        self._copy_raw_action = QtWidgets.QAction('Copy (Raw Text)', None)
        self._highlighter = FrontendHighlighter(self, lexer=self.lexer)
        self._kernel_manager = None
        self._kernel_client = None
        self._request_info = {}
        self._request_info['execute'] = {}
        self._callback_dict = {}
        self._display_banner = True
        self.tab_width = 4
        self._set_continuation_prompt('... ')
        self._call_tip_widget.setFont(self.font)
        self.font_changed.connect(self._call_tip_widget.setFont)
        action = self._copy_raw_action
        key = QtCore.Qt.CTRL | QtCore.Qt.SHIFT | QtCore.Qt.Key_C
        action.setEnabled(False)
        action.setShortcut(QtGui.QKeySequence(key))
        action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        action.triggered.connect(self.copy_raw)
        self.copy_available.connect(action.setEnabled)
        self.addAction(action)
        document = self._control.document()
        document.contentsChange.connect(self._document_contents_change)
        self._local_kernel = local_kernel
        self._pending_clearoutput = False

    def copy(self):
        if False:
            i = 10
            return i + 15
        ' Copy the currently selected text to the clipboard, removing prompts.\n        '
        if self._page_control is not None and self._page_control.hasFocus():
            self._page_control.copy()
        elif self._control.hasFocus():
            text = self._control.textCursor().selection().toPlainText()
            if text:
                (first_line_selection, *remaining_lines) = text.splitlines()
                cursor = self._control.textCursor()
                cursor.setPosition(cursor.selectionStart())
                cursor.setPosition(cursor.block().position(), QtGui.QTextCursor.KeepAnchor)
                preceding_text = cursor.selection().toPlainText()

                def remove_prompts(line):
                    if False:
                        i = 10
                        return i + 15
                    'Remove all prompts from line.'
                    line = self._highlighter.transform_classic_prompt(line)
                    return self._highlighter.transform_ipy_prompt(line)
                first_line = preceding_text + first_line_selection
                len_with_prompt = len(first_line)
                first_line = remove_prompts(first_line)
                prompt_len = len_with_prompt - len(first_line)
                if prompt_len < len(preceding_text):
                    first_line = first_line[len(preceding_text) - prompt_len:]
                if len(remaining_lines) > 0 and remaining_lines[-1]:
                    cursor = self._control.textCursor()
                    cursor.setPosition(cursor.selectionEnd())
                    block = cursor.block()
                    start_pos = block.position()
                    length = block.length()
                    cursor.setPosition(start_pos)
                    cursor.setPosition(start_pos + length - 1, QtGui.QTextCursor.KeepAnchor)
                    last_line_full = cursor.selection().toPlainText()
                    prompt_len = len(last_line_full) - len(remove_prompts(last_line_full))
                    if len(remaining_lines[-1]) < prompt_len:
                        remaining_lines[-1] = ''
                remaining_lines = map(remove_prompts, remaining_lines)
                text = '\n'.join([first_line, *remaining_lines])
                try:
                    was_newline = text[-1] == '\n'
                except IndexError:
                    was_newline = False
                if was_newline:
                    text = text[:-1]
                QtWidgets.QApplication.clipboard().setText(text)
        else:
            self.log.debug('frontend widget : unknown copy target')

    def _execute(self, source, hidden):
        if False:
            for i in range(10):
                print('nop')
        " Execute 'source'. If 'hidden', do not show any output.\n\n        See parent class :meth:`execute` docstring for full details.\n        "
        msg_id = self.kernel_client.execute(source, hidden)
        self._request_info['execute'][msg_id] = self._ExecutionRequest(msg_id, 'user', hidden)
        if not hidden:
            self.executing.emit(source)

    def _prompt_started_hook(self):
        if False:
            i = 10
            return i + 15
        ' Called immediately after a new prompt is displayed.\n        '
        if not self._reading:
            self._highlighter.highlighting_on = True

    def _prompt_finished_hook(self):
        if False:
            i = 10
            return i + 15
        ' Called immediately after a prompt is finished, i.e. when some input\n            will be processed and a new prompt displayed.\n        '
        if not self._reading:
            self._highlighter.highlighting_on = False

    def _tab_pressed(self):
        if False:
            print('Hello World!')
        ' Called when the tab key is pressed. Returns whether to continue\n            processing the event.\n        '
        text = self._get_input_buffer_cursor_line()
        if text is None:
            return False
        non_ws_before = bool(text[:self._get_input_buffer_cursor_column()].strip())
        complete = non_ws_before and self._get_cursor().selectedText() == ''
        if complete:
            self._complete()
        return not complete

    def _context_menu_make(self, pos):
        if False:
            return 10
        ' Reimplemented to add an action for raw copy.\n        '
        menu = super()._context_menu_make(pos)
        for before_action in menu.actions():
            if before_action.shortcut().matches(QtGui.QKeySequence.Paste) == QtGui.QKeySequence.ExactMatch:
                menu.insertAction(before_action, self._copy_raw_action)
                break
        return menu

    def request_interrupt_kernel(self):
        if False:
            return 10
        if self._executing:
            self.interrupt_kernel()

    def request_restart_kernel(self):
        if False:
            while True:
                i = 10
        message = 'Are you sure you want to restart the kernel?'
        self.restart_kernel(message, now=False)

    def _event_filter_console_keypress(self, event):
        if False:
            return 10
        ' Reimplemented for execution interruption and smart backspace.\n        '
        key = event.key()
        if self._control_key_down(event.modifiers(), include_command=False):
            if key == QtCore.Qt.Key_C and self._executing:
                if self.can_copy() and event.matches(QtGui.QKeySequence.Copy):
                    self.copy()
                else:
                    self.request_interrupt_kernel()
                return True
            elif key == QtCore.Qt.Key_Period:
                self.request_restart_kernel()
                return True
        elif not event.modifiers() & QtCore.Qt.AltModifier:
            if key == QtCore.Qt.Key_Backspace:
                col = self._get_input_buffer_cursor_column()
                cursor = self._control.textCursor()
                if col > 3 and (not cursor.hasSelection()):
                    text = self._get_input_buffer_cursor_line()[:col]
                    if text.endswith('    ') and (not text.strip()):
                        cursor.movePosition(QtGui.QTextCursor.Left, QtGui.QTextCursor.KeepAnchor, 4)
                        cursor.removeSelectedText()
                        return True
        return super()._event_filter_console_keypress(event)

    def _handle_clear_output(self, msg):
        if False:
            while True:
                i = 10
        'Handle clear output messages.'
        if self.include_output(msg):
            wait = msg['content'].get('wait', True)
            if wait:
                self._pending_clearoutput = True
            else:
                self.clear_output()

    def _silent_exec_callback(self, expr, callback):
        if False:
            return 10
        'Silently execute `expr` in the kernel and call `callback` with reply\n\n        the `expr` is evaluated silently in the kernel (without) output in\n        the frontend. Call `callback` with the\n        `repr <http://docs.python.org/library/functions.html#repr> `_ as first argument\n\n        Parameters\n        ----------\n        expr : string\n            valid string to be executed by the kernel.\n        callback : function\n            function accepting one argument, as a string. The string will be\n            the `repr` of the result of evaluating `expr`\n\n        The `callback` is called with the `repr()` of the result of `expr` as\n        first argument. To get the object, do `eval()` on the passed value.\n\n        See Also\n        --------\n        _handle_exec_callback : private method, deal with calling callback with reply\n\n        '
        local_uuid = str(uuid.uuid1())
        msg_id = self.kernel_client.execute('', silent=True, user_expressions={local_uuid: expr})
        self._callback_dict[local_uuid] = callback
        self._request_info['execute'][msg_id] = self._ExecutionRequest(msg_id, 'silent_exec_callback', False)

    def _handle_exec_callback(self, msg):
        if False:
            for i in range(10):
                print('nop')
        "Execute `callback` corresponding to `msg` reply, after ``_silent_exec_callback``\n\n        Parameters\n        ----------\n        msg : raw message send by the kernel containing an `user_expressions`\n                and having a 'silent_exec_callback' kind.\n\n        Notes\n        -----\n        This function will look for a `callback` associated with the\n        corresponding message id. Association has been made by\n        `_silent_exec_callback`. `callback` is then called with the `repr()`\n        of the value of corresponding `user_expressions` as argument.\n        `callback` is then removed from the known list so that any message\n        coming again with the same id won't trigger it.\n        "
        user_exp = msg['content'].get('user_expressions')
        if not user_exp:
            return
        for expression in user_exp:
            if expression in self._callback_dict:
                self._callback_dict.pop(expression)(user_exp[expression])

    def _handle_execute_reply(self, msg):
        if False:
            for i in range(10):
                print('nop')
        ' Handles replies for code execution.\n        '
        self.log.debug('execute_reply: %s', msg.get('content', ''))
        msg_id = msg['parent_header']['msg_id']
        info = self._request_info['execute'].get(msg_id)
        self._reading = False
        if not info or info.hidden:
            return
        if info.kind == 'user':
            if not self.kernel_client.iopub_channel.closed():
                self.kernel_client.iopub_channel.flush()
            if self.ansi_codes:
                self._ansi_processor.reset_sgr()
            content = msg['content']
            status = content['status']
            if status == 'ok':
                self._process_execute_ok(msg)
            elif status == 'aborted':
                self._process_execute_abort(msg)
            self._show_interpreter_prompt_for_reply(msg)
            self.executed.emit(msg)
            self._request_info['execute'].pop(msg_id)
        elif info.kind == 'silent_exec_callback':
            self._handle_exec_callback(msg)
            self._request_info['execute'].pop(msg_id)
        else:
            raise RuntimeError('Unknown handler for %s' % info.kind)

    def _handle_error(self, msg):
        if False:
            print('Hello World!')
        ' Handle error messages.\n        '
        self._process_execute_error(msg)

    def _handle_input_request(self, msg):
        if False:
            for i in range(10):
                print('nop')
        ' Handle requests for raw_input.\n        '
        self.log.debug('input: %s', msg.get('content', ''))
        msg_id = msg['parent_header']['msg_id']
        info = self._request_info['execute'].get(msg_id)
        if info and info.hidden:
            raise RuntimeError('Request for raw input during hidden execution.')
        if not self.kernel_client.iopub_channel.closed():
            self.kernel_client.iopub_channel.flush()

        def callback(line):
            if False:
                while True:
                    i = 10
            self._finalize_input_request()
            self.kernel_client.input(line)
        if self._reading:
            self.log.debug('Got second input request, assuming first was interrupted.')
            self._reading = False
        self._readline(msg['content']['prompt'], callback=callback, password=msg['content']['password'])

    def _kernel_restarted_message(self, died=True):
        if False:
            while True:
                i = 10
        msg = 'Kernel died, restarting' if died else 'Kernel restarting'
        self._append_html('<br>%s<hr><br>' % msg, before_prompt=False)

    def _handle_kernel_died(self, since_last_heartbeat):
        if False:
            for i in range(10):
                print('nop')
        "Handle the kernel's death (if we do not own the kernel).\n        "
        self.log.warning('kernel died: %s', since_last_heartbeat)
        if self.custom_restart:
            self.custom_restart_kernel_died.emit(since_last_heartbeat)
        else:
            self._kernel_restarted_message(died=True)
            self.reset()

    def _handle_kernel_restarted(self, died=True):
        if False:
            print('Hello World!')
        "Notice that the autorestarter restarted the kernel.\n\n        There's nothing to do but show a message.\n        "
        self.log.warning('kernel restarted')
        self._kernel_restarted_message(died=died)
        if self.kernel_manager:
            self.kernel_manager.reset_autorestart_count()
        self.reset()

    def _handle_inspect_reply(self, rep):
        if False:
            for i in range(10):
                print('nop')
        'Handle replies for call tips.'
        self.log.debug('oinfo: %s', rep.get('content', ''))
        cursor = self._get_cursor()
        info = self._request_info.get('call_tip')
        if info and info.id == rep['parent_header']['msg_id'] and (info.pos == cursor.position()):
            content = rep['content']
            if content.get('status') == 'ok' and content.get('found', False):
                self._call_tip_widget.show_inspect_data(content)

    def _handle_execute_result(self, msg):
        if False:
            print('Hello World!')
        ' Handle display hook output.\n        '
        self.log.debug('execute_result: %s', msg.get('content', ''))
        if self.include_output(msg):
            self.flush_clearoutput()
            text = msg['content']['data']
            self._append_plain_text(text + '\n', before_prompt=True)

    def _handle_stream(self, msg):
        if False:
            return 10
        ' Handle stdout, stderr, and stdin.\n        '
        self.log.debug('stream: %s', msg.get('content', ''))
        if self.include_output(msg):
            self.flush_clearoutput()
            self.append_stream(msg['content']['text'])

    def _handle_shutdown_reply(self, msg):
        if False:
            while True:
                i = 10
        ' Handle shutdown signal, only if from other console.\n        '
        self.log.debug('shutdown: %s', msg.get('content', ''))
        restart = msg.get('content', {}).get('restart', False)
        if msg['parent_header']:
            msg_id = msg['parent_header']['msg_id']
            info = self._request_info['execute'].get(msg_id)
            if info and info.hidden:
                return
        if not self.from_here(msg):
            if restart:
                self._handle_kernel_restarted(died=False)
            elif self._local_kernel:
                self.exit_requested.emit(self)
            else:
                title = self.window().windowTitle()
                reply = QtWidgets.QMessageBox.question(self, title, 'Kernel has been shutdown permanently. Close the Console?', QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
                if reply == QtWidgets.QMessageBox.Yes:
                    self.exit_requested.emit(self)

    def _handle_status(self, msg):
        if False:
            i = 10
            return i + 15
        'Handle status message'
        state = msg['content'].get('execution_state', '')
        if state == 'starting':
            if self._executing:
                self._handle_kernel_restarted(died=True)
        elif state == 'idle':
            pass
        elif state == 'busy':
            pass

    def _started_channels(self):
        if False:
            while True:
                i = 10
        ' Called when the KernelManager channels have started listening or\n            when the frontend is assigned an already listening KernelManager.\n        '
        self.reset(clear=True)

    def copy_raw(self):
        if False:
            i = 10
            return i + 15
        ' Copy the currently selected text to the clipboard without attempting\n            to remove prompts or otherwise alter the text.\n        '
        self._control.copy()

    def interrupt_kernel(self):
        if False:
            return 10
        ' Attempts to interrupt the running kernel.\n\n        Also unsets _reading flag, to avoid runtime errors\n        if raw_input is called again.\n        '
        if self.custom_interrupt:
            self._reading = False
            self.custom_interrupt_requested.emit()
        elif self.kernel_manager:
            self._reading = False
            self.kernel_manager.interrupt_kernel()
        else:
            self._append_plain_text('Cannot interrupt a kernel I did not start.\n')

    def reset(self, clear=False):
        if False:
            return 10
        ' Resets the widget to its initial state if ``clear`` parameter\n        is True, otherwise\n        prints a visual indication of the fact that the kernel restarted, but\n        does not clear the traces from previous usage of the kernel before it\n        was restarted.  With ``clear=True``, it is similar to ``%clear``, but\n        also re-writes the banner and aborts execution if necessary.\n        '
        if self._executing:
            self._executing = False
            self._request_info['execute'] = {}
        self._reading = False
        self._highlighter.highlighting_on = False
        if clear:
            self._control.clear()
            if self._display_banner:
                self._append_plain_text(self.banner)
                if self.kernel_banner:
                    self._append_plain_text(self.kernel_banner)
        self._show_interpreter_prompt()

    def restart_kernel(self, message, now=False):
        if False:
            for i in range(10):
                print('nop')
        ' Attempts to restart the running kernel.\n        '
        if self.custom_restart:
            self.custom_restart_requested.emit()
            return
        if self.kernel_manager:
            self.kernel_client.hb_channel.pause()
            if self.confirm_restart:
                buttons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                result = QtWidgets.QMessageBox.question(self, 'Restart kernel?', message, buttons)
                do_restart = result == QtWidgets.QMessageBox.Yes
            else:
                do_restart = True
            if do_restart:
                try:
                    self.kernel_manager.restart_kernel(now=now)
                except RuntimeError as e:
                    self._append_plain_text('Error restarting kernel: %s\n' % e, before_prompt=True)
                else:
                    self._append_html('<br>Restarting kernel...\n<hr><br>', before_prompt=True)
            else:
                self.kernel_client.hb_channel.unpause()
        else:
            self._append_plain_text('Cannot restart a Kernel I did not start\n', before_prompt=True)

    def append_stream(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Appends text to the output stream.'
        self._append_plain_text(text, before_prompt=True)

    def flush_clearoutput(self):
        if False:
            while True:
                i = 10
        'If a clearoutput is pending, execute it.'
        if self._pending_clearoutput:
            self._pending_clearoutput = False
            self.clear_output()

    def clear_output(self):
        if False:
            for i in range(10):
                print('nop')
        'Clears the current line of output.'
        cursor = self._control.textCursor()
        cursor.beginEditBlock()
        cursor.movePosition(QtGui.QTextCursor.StartOfLine, QtGui.QTextCursor.KeepAnchor)
        cursor.insertText('')
        cursor.endEditBlock()

    def _auto_call_tip(self):
        if False:
            return 10
        'Trigger call tip automatically on open parenthesis\n\n        Call tips can be requested explcitly with `_call_tip`.\n        '
        cursor = self._get_cursor()
        cursor.movePosition(QtGui.QTextCursor.Left)
        if cursor.document().characterAt(cursor.position()) == '(':
            self._call_tip()

    def _call_tip(self):
        if False:
            return 10
        'Shows a call tip, if appropriate, at the current cursor location.'
        if not self.enable_calltips or not self.kernel_client.shell_channel.is_alive():
            return False
        cursor_pos = self._get_input_buffer_cursor_pos()
        code = self.input_buffer
        msg_id = self.kernel_client.inspect(code, cursor_pos)
        pos = self._get_cursor().position()
        self._request_info['call_tip'] = self._CallTipRequest(msg_id, pos)
        return True

    def _complete(self):
        if False:
            return 10
        ' Performs completion at the current cursor location.\n        '
        code = self.input_buffer
        cursor_pos = self._get_input_buffer_cursor_pos()
        msg_id = self.kernel_client.complete(code=code, cursor_pos=cursor_pos)
        info = self._CompletionRequest(msg_id, code, cursor_pos)
        self._request_info['complete'] = info

    def _process_execute_abort(self, msg):
        if False:
            i = 10
            return i + 15
        ' Process a reply for an aborted execution request.\n        '
        self._append_plain_text('ERROR: execution aborted\n')

    def _process_execute_error(self, msg):
        if False:
            i = 10
            return i + 15
        ' Process a reply for an execution request that resulted in an error.\n        '
        content = msg['content']
        if content['ename'] == 'SystemExit':
            keepkernel = content['evalue'] == '-k' or content['evalue'] == 'True'
            self._keep_kernel_on_exit = keepkernel
            self.exit_requested.emit(self)
        else:
            traceback = ''.join(content['traceback'])
            self._append_plain_text(traceback)

    def _process_execute_ok(self, msg):
        if False:
            while True:
                i = 10
        ' Process a reply for a successful execution request.\n        '
        payload = msg['content'].get('payload', [])
        for item in payload:
            if not self._process_execute_payload(item):
                warning = 'Warning: received unknown payload of type %s'
                print(warning % repr(item['source']))

    def _process_execute_payload(self, item):
        if False:
            i = 10
            return i + 15
        ' Process a single payload item from the list of payload items in an\n            execution reply. Returns whether the payload was handled.\n        '
        return False

    def _show_interpreter_prompt(self):
        if False:
            i = 10
            return i + 15
        ' Shows a prompt for the interpreter.\n        '
        self._show_prompt('>>> ')

    def _show_interpreter_prompt_for_reply(self, msg):
        if False:
            while True:
                i = 10
        " Shows a prompt for the interpreter given an 'execute_reply' message.\n        "
        self._show_interpreter_prompt()

    def _document_contents_change(self, position, removed, added):
        if False:
            i = 10
            return i + 15
        " Called whenever the document's content changes. Display a call tip\n            if appropriate.\n        "
        position += added
        if position == self._get_cursor().position():
            self._auto_call_tip()

    @default('banner')
    def _banner_default(self):
        if False:
            i = 10
            return i + 15
        ' Returns the standard Python banner.\n        '
        banner = 'Python %s on %s\nType "help", "copyright", "credits" or "license" for more information.'
        return banner % (sys.version, sys.platform)