"""
Editor mixin and utils to manage connection with the LSP
"""
import functools
import logging
import random
import re
from diff_match_patch import diff_match_patch
from qtpy.QtCore import QEventLoop, Qt, QTimer, QThread, Signal, Slot
from qtpy.QtGui import QColor, QTextCursor
from three_merge import merge
from spyder.config.base import get_debug_level, running_under_pytest
from spyder.plugins.completion.api import CompletionRequestTypes, TextDocumentSyncKind, DiagnosticSeverity
from spyder.plugins.completion.decorators import request, handles, class_register
from spyder.plugins.editor.panels import FoldingPanel
from spyder.plugins.editor.panels.utils import merge_folding, collect_folding_regions
from spyder.plugins.editor.utils.editor import BlockUserData
from spyder.utils import sourcecode
logger = logging.getLogger(__name__)
NOQA_INLINE_REGEXP = re.compile('#?noqa', re.IGNORECASE)

def schedule_request(req=None, method=None, requires_response=True):
    if False:
        while True:
            i = 10
    'Call function req and then emit its results to the completion server.'
    if req is None:
        return functools.partial(schedule_request, method=method, requires_response=requires_response)

    @functools.wraps(req)
    def wrapper(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        params = req(self, *args, **kwargs)
        if params is not None and self.completions_available:
            self._pending_server_requests.append((method, params, requires_response))
            self._server_requests_timer.setInterval(self.LSP_REQUESTS_SHORT_DELAY)
            self._server_requests_timer.start()
    return wrapper

@class_register
class LSPMixin:
    SYNC_SYMBOLS_AND_FOLDING_TIMEOUTS = {500: 600, 1500: 800, 2500: 1000, 6500: 1500}
    LSP_REQUESTS_SHORT_DELAY = 50
    LSP_REQUESTS_LONG_DELAY = 300
    sig_perform_completion_request = Signal(str, str, dict)
    completions_response_signal = Signal(str, object)
    sig_display_object_info = Signal(str, bool)
    sig_signature_invoked = Signal(dict)
    sig_process_code_analysis = Signal()
    sig_start_operation_in_progress = Signal()
    sig_stop_operation_in_progress = Signal()

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self._timer_sync_symbols_and_folding = QTimer(self)
        self._timer_sync_symbols_and_folding.setSingleShot(True)
        self.blockCountChanged.connect(self.set_sync_symbols_and_folding_timeout)
        self._pending_server_requests = []
        self._server_requests_timer = QTimer(self)
        self._server_requests_timer.setSingleShot(True)
        self._server_requests_timer.setInterval(self.LSP_REQUESTS_SHORT_DELAY)
        self._server_requests_timer.timeout.connect(self._process_server_requests)
        self.code_folding = True
        self.update_folding_thread = QThread(None)
        self.update_folding_thread.finished.connect(self.finish_code_folding)
        self.format_on_save = False
        self.format_eventloop = QEventLoop(None)
        self.format_timer = QTimer(self)
        self.__cursor_position_before_format = 0
        self.oe_proxy = None
        self.update_diagnostics_thread = QThread(None)
        self.update_diagnostics_thread.run = self.set_errors
        self.update_diagnostics_thread.finished.connect(self.finish_code_analysis)
        self._diagnostics = []
        self.differ = diff_match_patch()
        self.previous_text = ''
        self.patch = []
        self.leading_whitespaces = {}
        self.filename = None
        self.completions_available = False
        self.text_version = 0
        self.save_include_text = True
        self.open_close_notifications = True
        self.sync_mode = TextDocumentSyncKind.FULL
        self.will_save_notify = False
        self.will_save_until_notify = False
        self.enable_hover = True
        self.auto_completion_characters = []
        self.resolve_completions_enabled = False
        self.signature_completion_characters = []
        self.go_to_definition_enabled = False
        self.find_references_enabled = False
        self.highlight_enabled = False
        self.formatting_enabled = False
        self.range_formatting_enabled = False
        self.document_symbols_enabled = False
        self.formatting_characters = []
        self.completion_args = None
        self.folding_supported = False
        self._folding_info = None
        self.is_cloned = False
        self.operation_in_progress = False
        self.formatting_in_progress = False
        self.symbols_in_sync = False
        self.folding_in_sync = False

    def _process_server_requests(self):
        if False:
            return 10
        'Process server requests.'
        if self._document_server_needs_update:
            self.document_did_change()
            self.do_automatic_completions()
            self._document_server_needs_update = False
        for (method, params, requires_response) in self._pending_server_requests:
            self.emit_request(method, params, requires_response)
        self._pending_server_requests = []

    @Slot(str, dict)
    def handle_response(self, method, params):
        if False:
            while True:
                i = 10
        if method in self.handler_registry:
            handler_name = self.handler_registry[method]
            handler = getattr(self, handler_name)
            handler(params)
            self.completions_response_signal.emit(method, params)

    def emit_request(self, method, params, requires_response):
        if False:
            print('Hello World!')
        'Send request to LSP manager.'
        params['requires_response'] = requires_response
        params['response_instance'] = self
        self.sig_perform_completion_request.emit(self.language.lower(), method, params)

    def log_lsp_handle_errors(self, message):
        if False:
            return 10
        '\n        Log errors when handling LSP responses.\n\n        This works when debugging is on or off.\n        '
        if get_debug_level() > 0:
            logger.error(message, exc_info=True)
        else:
            logger.error('%', 1, stack_info=True)

    def start_completion_services(self):
        if False:
            i = 10
            return i + 15
        'Start completion services for this instance.'
        self.completions_available = True
        if self.is_cloned:
            additional_msg = 'cloned editor'
        else:
            additional_msg = ''
            self.document_did_open()
        logger.debug('Completion services available for {0}: {1}'.format(additional_msg, self.filename))

    def register_completion_capabilities(self, capabilities):
        if False:
            for i in range(10):
                print('nop')
        '\n        Register completion server capabilities.\n\n        Parameters\n        ----------\n        capabilities: dict\n            Capabilities supported by a language server.\n        '
        sync_options = capabilities['textDocumentSync']
        completion_options = capabilities['completionProvider']
        signature_options = capabilities['signatureHelpProvider']
        range_formatting_options = capabilities['documentOnTypeFormattingProvider']
        self.open_close_notifications = sync_options.get('openClose', False)
        self.sync_mode = sync_options.get('change', TextDocumentSyncKind.NONE)
        self.will_save_notify = sync_options.get('willSave', False)
        self.will_save_until_notify = sync_options.get('willSaveWaitUntil', False)
        self.save_include_text = sync_options['save']['includeText']
        self.enable_hover = capabilities['hoverProvider']
        self.folding_supported = capabilities.get('foldingRangeProvider', False)
        self.auto_completion_characters = completion_options['triggerCharacters']
        self.resolve_completions_enabled = completion_options.get('resolveProvider', False)
        self.signature_completion_characters = signature_options['triggerCharacters'] + ['=']
        self.go_to_definition_enabled = capabilities['definitionProvider']
        self.find_references_enabled = capabilities['referencesProvider']
        self.highlight_enabled = capabilities['documentHighlightProvider']
        self.formatting_enabled = capabilities['documentFormattingProvider']
        self.range_formatting_enabled = capabilities['documentRangeFormattingProvider']
        self.document_symbols_enabled = capabilities['documentSymbolProvider']
        self.formatting_characters.append(range_formatting_options['firstTriggerCharacter'])
        self.formatting_characters += range_formatting_options.get('moreTriggerCharacter', [])
        if self.formatting_enabled:
            self.format_action.setEnabled(True)
            self.sig_refresh_formatting.emit(True)
        self.completions_available = True

    def stop_completion_services(self):
        if False:
            i = 10
            return i + 15
        logger.debug('Stopping completion services for %s' % self.filename)
        self.completions_available = False

    @request(method=CompletionRequestTypes.DOCUMENT_DID_OPEN, requires_response=False)
    def document_did_open(self):
        if False:
            while True:
                i = 10
        'Send textDocument/didOpen request to the server.'
        try:
            self._timer_sync_symbols_and_folding.timeout.disconnect()
        except (TypeError, RuntimeError):
            pass
        self._timer_sync_symbols_and_folding.timeout.connect(self.sync_symbols_and_folding, Qt.UniqueConnection)
        cursor = self.textCursor()
        text = self.get_text_with_eol()
        if self.is_ipython():
            text = self.ipython_to_python(text)
        params = {'file': self.filename, 'language': self.language, 'version': self.text_version, 'text': text, 'codeeditor': self, 'offset': cursor.position(), 'selection_start': cursor.selectionStart(), 'selection_end': cursor.selectionEnd()}
        return params

    @schedule_request(method=CompletionRequestTypes.DOCUMENT_SYMBOL)
    def request_symbols(self):
        if False:
            return 10
        'Request document symbols.'
        if not self.document_symbols_enabled:
            return
        if self.oe_proxy is not None:
            self.oe_proxy.emit_request_in_progress()
        params = {'file': self.filename}
        return params

    @handles(CompletionRequestTypes.DOCUMENT_SYMBOL)
    def process_symbols(self, params):
        if False:
            return 10
        'Handle symbols response.'
        try:
            symbols = params['params']
            self._update_classfuncdropdown(symbols)
            if self.oe_proxy is not None:
                self.oe_proxy.update_outline_info(symbols)
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when processing symbols')
        finally:
            self.symbols_in_sync = True

    def _update_classfuncdropdown(self, symbols):
        if False:
            while True:
                i = 10
        'Update class/function dropdown.'
        symbols = [] if symbols is None else symbols
        if self.classfuncdropdown.isVisible():
            self.classfuncdropdown.update_data(symbols)
        else:
            self.classfuncdropdown.set_data(symbols)

    def _schedule_document_did_change(self):
        if False:
            return 10
        'Schedule a document update.'
        self._document_server_needs_update = True
        self._server_requests_timer.setInterval(self.LSP_REQUESTS_LONG_DELAY)
        self._server_requests_timer.start()

    @request(method=CompletionRequestTypes.DOCUMENT_DID_CHANGE, requires_response=False)
    def document_did_change(self):
        if False:
            for i in range(10):
                print('nop')
        'Send textDocument/didChange request to the server.'
        self.formatting_in_progress = False
        self.symbols_in_sync = False
        self.folding_in_sync = False
        if self.is_cloned:
            return
        text = self.get_text_with_eol()
        if self.is_ipython():
            text = self.ipython_to_python(text)
        self.text_version += 1
        self.patch = self.differ.patch_make(self.previous_text, text)
        self.previous_text = text
        cursor = self.textCursor()
        params = {'file': self.filename, 'version': self.text_version, 'text': text, 'diff': self.patch, 'offset': cursor.position(), 'selection_start': cursor.selectionStart(), 'selection_end': cursor.selectionEnd()}
        return params

    @handles(CompletionRequestTypes.DOCUMENT_PUBLISH_DIAGNOSTICS)
    def process_diagnostics(self, params):
        if False:
            print('Hello World!')
        'Handle linting response.'
        self._timer_sync_symbols_and_folding.start()
        self.process_code_analysis(params['params'])

    def set_sync_symbols_and_folding_timeout(self):
        if False:
            print('Hello World!')
        '\n        Set timeout to sync symbols and folding according to the file\n        size.\n        '
        current_lines = self.get_line_count()
        timeout = None
        for lines in self.SYNC_SYMBOLS_AND_FOLDING_TIMEOUTS.keys():
            if current_lines // lines == 0:
                timeout = self.SYNC_SYMBOLS_AND_FOLDING_TIMEOUTS[lines]
                break
        if not timeout:
            timeouts = self.SYNC_SYMBOLS_AND_FOLDING_TIMEOUTS.values()
            timeout = list(timeouts)[-1]
        self._timer_sync_symbols_and_folding.setInterval(timeout + random.randint(-100, 100))

    def sync_symbols_and_folding(self):
        if False:
            print('Hello World!')
        '\n        Synchronize symbols and folding after linting results arrive.\n        '
        if not self.folding_in_sync:
            self.request_folding()
        if not self.symbols_in_sync:
            self.request_symbols()

    def process_code_analysis(self, diagnostics):
        if False:
            print('Hello World!')
        'Process code analysis results in a thread.'
        self.cleanup_code_analysis()
        self._diagnostics = diagnostics
        self.update_diagnostics_thread.start()

    def cleanup_code_analysis(self):
        if False:
            return 10
        'Remove all code analysis markers'
        self.setUpdatesEnabled(False)
        self.clear_extra_selections('code_analysis_highlight')
        self.clear_extra_selections('code_analysis_underline')
        for data in self.blockuserdata_list():
            data.code_analysis = []
        self.setUpdatesEnabled(True)
        self.sig_flags_changed.emit()
        self.linenumberarea.update()

    def set_errors(self):
        if False:
            print('Hello World!')
        'Set errors and warnings in the line number area.'
        try:
            self._process_code_analysis(underline=False)
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when processing linting')

    def underline_errors(self):
        if False:
            i = 10
            return i + 15
        'Underline errors and warnings.'
        try:
            self.clear_extra_selections('code_analysis_underline')
            self._process_code_analysis(underline=True)
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when processing linting')

    def finish_code_analysis(self):
        if False:
            i = 10
            return i + 15
        'Finish processing code analysis results.'
        self.linenumberarea.update()
        if self.underline_errors_enabled:
            self.underline_errors()
        self.sig_process_code_analysis.emit()
        self.sig_flags_changed.emit()

    def errors_present(self):
        if False:
            while True:
                i = 10
        '\n        Return True if there are errors or warnings present in the file.\n        '
        return bool(len(self._diagnostics))

    def _process_code_analysis(self, underline):
        if False:
            i = 10
            return i + 15
        "\n        Process all code analysis results.\n\n        Parameters\n        ----------\n        underline: bool\n            Determines if errors and warnings are going to be set in\n            the line number area or underlined. It's better to separate\n            these two processes for perfomance reasons. That's because\n            setting errors can be done in a thread whereas underlining\n            them can't.\n        "
        document = self.document()
        if underline:
            (first_block, last_block) = self.get_buffer_block_numbers()
        for diagnostic in self._diagnostics:
            if self.is_ipython() and diagnostic['message'] == "undefined name 'get_ipython'":
                continue
            source = diagnostic.get('source', '')
            msg_range = diagnostic['range']
            start = msg_range['start']
            end = msg_range['end']
            code = diagnostic.get('code', 'E')
            message = diagnostic['message']
            severity = diagnostic.get('severity', DiagnosticSeverity.ERROR)
            block = document.findBlockByNumber(start['line'])
            text = block.text()
            if 'analysis:ignore' in text:
                continue
            if self.language == 'Python':
                if NOQA_INLINE_REGEXP.search(text) is not None:
                    continue
            data = block.userData()
            if not data:
                data = BlockUserData(self)
            if underline:
                block_nb = block.blockNumber()
                if first_block <= block_nb <= last_block:
                    error = severity == DiagnosticSeverity.ERROR
                    color = self.error_color if error else self.warning_color
                    color = QColor(color)
                    color.setAlpha(255)
                    block.color = color
                    data.selection_start = start
                    data.selection_end = end
                    self.highlight_selection('code_analysis_underline', data._selection(), underline_color=block.color)
            else:
                if not self.is_cloned:
                    data.code_analysis.append((source, code, severity, message))
                block.setUserData(data)

    @schedule_request(method=CompletionRequestTypes.DOCUMENT_COMPLETION)
    def do_completion(self, automatic=False):
        if False:
            print('Hello World!')
        'Trigger completion.'
        cursor = self.textCursor()
        current_word = self.get_current_word(completion=True, valid_python_variable=False)
        params = {'file': self.filename, 'line': cursor.blockNumber(), 'column': cursor.columnNumber(), 'offset': cursor.position(), 'selection_start': cursor.selectionStart(), 'selection_end': cursor.selectionEnd(), 'current_word': current_word}
        self.completion_args = (self.textCursor().position(), automatic)
        return params

    @handles(CompletionRequestTypes.DOCUMENT_COMPLETION)
    def process_completion(self, params):
        if False:
            for i in range(10):
                print('nop')
        'Handle completion response.'
        args = self.completion_args
        if args is None:
            return
        self.completion_args = None
        (position, automatic) = args
        start_cursor = self.textCursor()
        start_cursor.movePosition(QTextCursor.StartOfBlock)
        line_text = self.get_text(start_cursor.position(), 'eol')
        leading_whitespace = self.compute_whitespace(line_text)
        indentation_whitespace = ' ' * leading_whitespace
        eol_char = self.get_line_separator()
        try:
            completions = params['params']
            completions = [] if completions is None else [completion for completion in completions if completion.get('insertText') or completion.get('textEdit', {}).get('newText')]
            prefix = self.get_current_word(completion=True, valid_python_variable=False)
            if len(completions) == 1 and completions[0].get('insertText') == prefix and (not completions[0].get('textEdit', {}).get('newText')):
                completions.pop()
            replace_end = self.textCursor().position()
            under_cursor = self.get_current_word_and_position(completion=True)
            if under_cursor:
                (word, replace_start) = under_cursor
            else:
                word = ''
                replace_start = replace_end
            first_letter = ''
            if len(word) > 0:
                first_letter = word[0]

            def sort_key(completion):
                if False:
                    print('Hello World!')
                if 'textEdit' in completion:
                    text_insertion = completion['textEdit']['newText']
                else:
                    text_insertion = completion['insertText']
                first_insert_letter = text_insertion[0]
                case_mismatch = first_letter.isupper() and first_insert_letter.islower() or (first_letter.islower() and first_insert_letter.isupper())
                return (case_mismatch, completion['sortText'])
            completion_list = sorted(completions, key=sort_key)
            for completion in completion_list:
                if 'textEdit' in completion:
                    c_replace_start = completion['textEdit']['range']['start']
                    c_replace_end = completion['textEdit']['range']['end']
                    if c_replace_start == replace_start and c_replace_end == replace_end:
                        insert_text = completion['textEdit']['newText']
                        completion['filterText'] = insert_text
                        completion['insertText'] = insert_text
                        del completion['textEdit']
                if 'insertText' in completion:
                    insert_text = completion['insertText']
                    insert_text_lines = insert_text.splitlines()
                    reindented_text = [insert_text_lines[0]]
                    for insert_line in insert_text_lines[1:]:
                        insert_line = indentation_whitespace + insert_line
                        reindented_text.append(insert_line)
                    reindented_text = eol_char.join(reindented_text)
                    completion['insertText'] = reindented_text
            self.completion_widget.show_list(completion_list, position, automatic)
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when processing completions')

    @schedule_request(method=CompletionRequestTypes.COMPLETION_RESOLVE)
    def resolve_completion_item(self, item):
        if False:
            return 10
        return {'file': self.filename, 'completion_item': item}

    @handles(CompletionRequestTypes.COMPLETION_RESOLVE)
    def handle_completion_item_resolution(self, response):
        if False:
            while True:
                i = 10
        try:
            response = response['params']
            if not response:
                return
            self.completion_widget.augment_completion_info(response)
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when handling completion item resolution')

    @schedule_request(method=CompletionRequestTypes.DOCUMENT_SIGNATURE)
    def request_signature(self):
        if False:
            return 10
        'Ask for signature.'
        (line, column) = self.get_cursor_line_column()
        offset = self.get_position('cursor')
        params = {'file': self.filename, 'line': line, 'column': column, 'offset': offset}
        return params

    @handles(CompletionRequestTypes.DOCUMENT_SIGNATURE)
    def process_signatures(self, params):
        if False:
            return 10
        'Handle signature response.'
        try:
            signature_params = params['params']
            if signature_params is not None and 'activeParameter' in signature_params:
                self.sig_signature_invoked.emit(signature_params)
                signature_data = signature_params['signatures']
                documentation = signature_data['documentation']
                if isinstance(documentation, dict):
                    documentation = documentation['value']
                documentation = documentation.replace('\xa0', ' ')
                parameter_idx = signature_params['activeParameter']
                parameters = signature_data['parameters']
                parameter = None
                if len(parameters) > 0 and parameter_idx < len(parameters):
                    parameter_data = parameters[parameter_idx]
                    parameter = parameter_data['label']
                signature = signature_data['label']
                self.show_calltip(signature=signature, parameter=parameter, language=self.language, documentation=documentation)
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when processing signature')

    @schedule_request(method=CompletionRequestTypes.DOCUMENT_CURSOR_EVENT)
    def request_cursor_event(self):
        if False:
            while True:
                i = 10
        text = self.get_text_with_eol()
        cursor = self.textCursor()
        params = {'file': self.filename, 'version': self.text_version, 'text': text, 'offset': cursor.position(), 'selection_start': cursor.selectionStart(), 'selection_end': cursor.selectionEnd()}
        return params

    @schedule_request(method=CompletionRequestTypes.DOCUMENT_HOVER)
    def request_hover(self, line, col, offset, show_hint=True, clicked=True):
        if False:
            while True:
                i = 10
        'Request hover information.'
        params = {'file': self.filename, 'line': line, 'column': col, 'offset': offset}
        self._show_hint = show_hint
        self._request_hover_clicked = clicked
        return params

    @handles(CompletionRequestTypes.DOCUMENT_HOVER)
    def handle_hover_response(self, contents):
        if False:
            while True:
                i = 10
        'Handle hover response.'
        if running_under_pytest():
            from unittest.mock import Mock
            if isinstance(contents, Mock):
                return
        try:
            content = contents['params']
            if not content or isinstance(content, list):
                return
            self.sig_display_object_info.emit(content, self._request_hover_clicked)
            if content is not None and self._show_hint and self._last_point:
                word = self._last_hover_word
                content = content.replace('\xa0', ' ')
                self.show_hint(content, inspect_word=word, at_point=self._last_point)
                self._last_point = None
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when processing hover')

    @Slot()
    @schedule_request(method=CompletionRequestTypes.DOCUMENT_DEFINITION)
    def go_to_definition_from_cursor(self, cursor=None):
        if False:
            while True:
                i = 10
        'Go to definition from cursor instance (QTextCursor).'
        if not self.go_to_definition_enabled or self.in_comment_or_string():
            return
        if cursor is None:
            cursor = self.textCursor()
        text = str(cursor.selectedText())
        if len(text) == 0:
            cursor.select(QTextCursor.WordUnderCursor)
            text = str(cursor.selectedText())
        if text is not None:
            (line, column) = self.get_cursor_line_column()
            params = {'file': self.filename, 'line': line, 'column': column}
            return params

    @handles(CompletionRequestTypes.DOCUMENT_DEFINITION)
    def handle_go_to_definition(self, position):
        if False:
            i = 10
            return i + 15
        'Handle go to definition response.'
        try:
            position = position['params']
            if position is not None:
                def_range = position['range']
                start = def_range['start']
                if self.filename == position['file']:
                    self.go_to_line(start['line'] + 1, start['character'], None, word=None)
                else:
                    self.go_to_definition.emit(position['file'], start['line'] + 1, start['character'])
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when processing go to definition')

    def format_document_or_range(self):
        if False:
            print('Hello World!')
        'Format current document or selected text.'
        if self.has_selected_text() and self.range_formatting_enabled:
            self.format_document_range()
        else:
            self.format_document()

    @schedule_request(method=CompletionRequestTypes.DOCUMENT_FORMATTING)
    def format_document(self):
        if False:
            for i in range(10):
                print('nop')
        'Format current document.'
        self.__cursor_position_before_format = self.textCursor().position()
        if not self.formatting_enabled:
            return
        if self.formatting_in_progress:
            return
        using_spaces = self.indent_chars != '\t'
        tab_size = len(self.indent_chars) if using_spaces else self.tab_stop_width_spaces
        params = {'file': self.filename, 'options': {'tab_size': tab_size, 'insert_spaces': using_spaces, 'trim_trailing_whitespace': self.remove_trailing_spaces, 'insert_final_new_line': self.add_newline, 'trim_final_new_lines': self.remove_trailing_newlines}}
        self.setReadOnly(True)
        self.document().setModified(True)
        self.sig_start_operation_in_progress.emit()
        self.operation_in_progress = True
        self.formatting_in_progress = True
        return params

    @schedule_request(method=CompletionRequestTypes.DOCUMENT_RANGE_FORMATTING)
    def format_document_range(self):
        if False:
            return 10
        'Format selected text.'
        self.__cursor_position_before_format = self.textCursor().position()
        if not self.range_formatting_enabled or not self.has_selected_text():
            return
        if self.formatting_in_progress:
            return
        (start, end) = self.get_selection_start_end()
        (start_line, start_col) = start
        (end_line, end_col) = end
        using_spaces = self.indent_chars != '\t'
        tab_size = len(self.indent_chars) if using_spaces else self.tab_stop_width_spaces
        fmt_range = {'start': {'line': start_line, 'character': start_col}, 'end': {'line': end_line, 'character': end_col}}
        params = {'file': self.filename, 'range': fmt_range, 'options': {'tab_size': tab_size, 'insert_spaces': using_spaces, 'trim_trailing_whitespace': self.remove_trailing_spaces, 'insert_final_new_line': self.add_newline, 'trim_final_new_lines': self.remove_trailing_newlines}}
        self.setReadOnly(True)
        self.document().setModified(True)
        self.sig_start_operation_in_progress.emit()
        self.operation_in_progress = True
        self.formatting_in_progress = True
        return params

    @handles(CompletionRequestTypes.DOCUMENT_FORMATTING)
    def handle_document_formatting(self, edits):
        if False:
            print('Hello World!')
        'Handle document formatting response.'
        try:
            if self.formatting_in_progress:
                self._apply_document_edits(edits)
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when processing document formatting')
        finally:
            self.setReadOnly(False)
            self.document().setModified(False)
            self.document().setModified(True)
            self.sig_stop_operation_in_progress.emit()
            self.operation_in_progress = False
            self.formatting_in_progress = False

    @handles(CompletionRequestTypes.DOCUMENT_RANGE_FORMATTING)
    def handle_document_range_formatting(self, edits):
        if False:
            print('Hello World!')
        'Handle document range formatting response.'
        try:
            if self.formatting_in_progress:
                self._apply_document_edits(edits)
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when processing document selection formatting')
        finally:
            self.setReadOnly(False)
            self.document().setModified(False)
            self.document().setModified(True)
            self.sig_stop_operation_in_progress.emit()
            self.operation_in_progress = False
            self.formatting_in_progress = False

    def _apply_document_edits(self, edits):
        if False:
            return 10
        'Apply a set of atomic document edits to the current editor text.'
        edits = edits['params']
        if edits is None:
            return
        text = self.toPlainText()
        text_tokens = list(text)
        merged_text = None
        for edit in edits:
            edit_range = edit['range']
            repl_text = edit['newText']
            (start, end) = (edit_range['start'], edit_range['end'])
            (start_line, start_col) = (start['line'], start['character'])
            (end_line, end_col) = (end['line'], end['character'])
            start_pos = self.get_position_line_number(start_line, start_col)
            end_pos = self.get_position_line_number(end_line, end_col)
            repl_eol = sourcecode.get_eol_chars(repl_text)
            if repl_eol is not None and repl_eol != '\n':
                repl_text = repl_text.replace(repl_eol, '\n')
            text_tokens = list(text_tokens)
            this_edit = list(repl_text)
            if end_line == self.document().blockCount():
                end_pos = self.get_position('eof')
                end_pos += 1
            if end_pos == len(text_tokens) and text_tokens[end_pos - 1] == '\n':
                end_pos += 1
            this_edition = text_tokens[:max(start_pos - 1, 0)] + this_edit + text_tokens[end_pos - 1:]
            text_edit = ''.join(this_edition)
            if merged_text is None:
                merged_text = text_edit
            else:
                merged_text = merge(text_edit, merged_text, text)
        if merged_text is not None:
            merged_text = merged_text.replace('\n', self.get_line_separator())
            cursor = self.textCursor()
            cursor.beginEditBlock()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            cursor.insertText(merged_text)
            cursor.endEditBlock()
            if self.__cursor_position_before_format:
                self.moveCursor(QTextCursor.Start)
                cursor = self.textCursor()
                cursor.setPosition(self.__cursor_position_before_format)
                self.setTextCursor(cursor)
                self.centerCursor()

    def compute_whitespace(self, line):
        if False:
            for i in range(10):
                print('nop')
        tab_size = self.tab_stop_width_spaces
        whitespace_regex = re.compile('(\\s+).*')
        whitespace_match = whitespace_regex.match(line)
        total_whitespace = 0
        if whitespace_match is not None:
            whitespace_chars = whitespace_match.group(1)
            whitespace_chars = whitespace_chars.replace('\t', tab_size * ' ')
            total_whitespace = len(whitespace_chars)
        return total_whitespace

    def update_whitespace_count(self, line, column):
        if False:
            print('Hello World!')
        self.leading_whitespaces = {}
        lines = str(self.toPlainText()).splitlines()
        for (i, text) in enumerate(lines):
            total_whitespace = self.compute_whitespace(text)
            self.leading_whitespaces[i] = total_whitespace

    def cleanup_folding(self):
        if False:
            while True:
                i = 10
        'Cleanup folding pane.'
        folding_panel = self.panels.get(FoldingPanel)
        folding_panel.folding_regions = {}

    @schedule_request(method=CompletionRequestTypes.DOCUMENT_FOLDING_RANGE)
    def request_folding(self):
        if False:
            while True:
                i = 10
        'Request folding.'
        if not self.folding_supported or not self.code_folding:
            return
        params = {'file': self.filename}
        return params

    @handles(CompletionRequestTypes.DOCUMENT_FOLDING_RANGE)
    def handle_folding_range(self, response):
        if False:
            for i in range(10):
                print('nop')
        'Handle folding response.'
        ranges = response['params']
        if ranges is None:
            return
        try:
            extended_ranges = []
            for (start, end) in ranges:
                text_region = self.get_text_region(start, end)
                extended_ranges.append((start, end, text_region))
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when processing folding')
        finally:
            self.folding_in_sync = True
        self.update_folding_thread.run = functools.partial(self.update_and_merge_folding, extended_ranges)
        self.update_folding_thread.start()

    def update_and_merge_folding(self, extended_ranges):
        if False:
            for i in range(10):
                print('nop')
        'Update and merge new folding information.'
        try:
            folding_panel = self.panels.get(FoldingPanel)
            (current_tree, root) = merge_folding(extended_ranges, folding_panel.current_tree, folding_panel.root)
            folding_info = collect_folding_regions(root)
            self._folding_info = (current_tree, root, *folding_info)
        except RuntimeError:
            return
        except Exception:
            self.log_lsp_handle_errors('Error when processing folding')

    def finish_code_folding(self):
        if False:
            while True:
                i = 10
        'Finish processing code folding.'
        folding_panel = self.panels.get(FoldingPanel)
        if self._folding_info is not None:
            folding_panel.update_folding(self._folding_info)
        if self.indent_guides._enabled and len(self.patch) > 0:
            (line, column) = self.get_cursor_line_column()
            self.update_whitespace_count(line, column)

    @schedule_request(method=CompletionRequestTypes.DOCUMENT_DID_SAVE, requires_response=False)
    def notify_save(self):
        if False:
            return 10
        'Send save request.'
        params = {'file': self.filename}
        if self.save_include_text:
            params['text'] = self.get_text_with_eol()
        return params

    @request(method=CompletionRequestTypes.DOCUMENT_DID_CLOSE, requires_response=False)
    def notify_close(self):
        if False:
            while True:
                i = 10
        'Send close request.'
        self._pending_server_requests = []
        try:
            self._server_requests_timer.stop()
        except RuntimeError:
            pass
        if self.completions_available:
            try:
                self._timer_sync_symbols_and_folding.timeout.disconnect()
            except (TypeError, RuntimeError):
                pass
            params = {'file': self.filename, 'codeeditor': self}
            return params