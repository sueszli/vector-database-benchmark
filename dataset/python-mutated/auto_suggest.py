import re
import tokenize
from io import StringIO
from typing import Callable, List, Optional, Union, Generator, Tuple
import warnings
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands as nc
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.history import History
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.layout.processors import Processor, Transformation, TransformationInput
from IPython.core.getipython import get_ipython
from IPython.utils.tokenutil import generate_tokens
from .filters import pass_through

def _get_query(document: Document):
    if False:
        while True:
            i = 10
    return document.lines[document.cursor_position_row]

class AppendAutoSuggestionInAnyLine(Processor):
    """
    Append the auto suggestion to lines other than the last (appending to the
    last line is natively supported by the prompt toolkit).
    """

    def __init__(self, style: str='class:auto-suggestion') -> None:
        if False:
            for i in range(10):
                print('nop')
        self.style = style

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        if False:
            while True:
                i = 10
        is_last_line = ti.lineno == ti.document.line_count - 1
        is_active_line = ti.lineno == ti.document.cursor_position_row
        if not is_last_line and is_active_line:
            buffer = ti.buffer_control.buffer
            if buffer.suggestion and ti.document.is_cursor_at_the_end_of_line:
                suggestion = buffer.suggestion.text
            else:
                suggestion = ''
            return Transformation(fragments=ti.fragments + [(self.style, suggestion)])
        else:
            return Transformation(fragments=ti.fragments)

class NavigableAutoSuggestFromHistory(AutoSuggestFromHistory):
    """
    A subclass of AutoSuggestFromHistory that allow navigation to next/previous
    suggestion from history. To do so it remembers the current position, but it
    state need to carefully be cleared on the right events.
    """

    def __init__(self):
        if False:
            return 10
        self.skip_lines = 0
        self._connected_apps = []

    def reset_history_position(self, _: Buffer):
        if False:
            while True:
                i = 10
        self.skip_lines = 0

    def disconnect(self):
        if False:
            return 10
        for pt_app in self._connected_apps:
            text_insert_event = pt_app.default_buffer.on_text_insert
            text_insert_event.remove_handler(self.reset_history_position)

    def connect(self, pt_app: PromptSession):
        if False:
            while True:
                i = 10
        self._connected_apps.append(pt_app)
        pt_app.default_buffer.on_text_insert.add_handler(self.reset_history_position)
        pt_app.default_buffer.on_cursor_position_changed.add_handler(self._dismiss)

    def get_suggestion(self, buffer: Buffer, document: Document) -> Optional[Suggestion]:
        if False:
            print('Hello World!')
        text = _get_query(document)
        if text.strip():
            for (suggestion, _) in self._find_next_match(text, self.skip_lines, buffer.history):
                return Suggestion(suggestion)
        return None

    def _dismiss(self, buffer, *args, **kwargs):
        if False:
            print('Hello World!')
        buffer.suggestion = None

    def _find_match(self, text: str, skip_lines: float, history: History, previous: bool) -> Generator[Tuple[str, float], None, None]:
        if False:
            while True:
                i = 10
        '\n        text : str\n            Text content to find a match for, the user cursor is most of the\n            time at the end of this text.\n        skip_lines : float\n            number of items to skip in the search, this is used to indicate how\n            far in the list the user has navigated by pressing up or down.\n            The float type is used as the base value is +inf\n        history : History\n            prompt_toolkit History instance to fetch previous entries from.\n        previous : bool\n            Direction of the search, whether we are looking previous match\n            (True), or next match (False).\n\n        Yields\n        ------\n        Tuple with:\n        str:\n            current suggestion.\n        float:\n            will actually yield only ints, which is passed back via skip_lines,\n            which may be a +inf (float)\n\n\n        '
        line_number = -1
        for string in reversed(list(history.get_strings())):
            for line in reversed(string.splitlines()):
                line_number += 1
                if not previous and line_number < skip_lines:
                    continue
                if line.startswith(text) and len(line) > len(text):
                    yield (line[len(text):], line_number)
                if previous and line_number >= skip_lines:
                    return

    def _find_next_match(self, text: str, skip_lines: float, history: History) -> Generator[Tuple[str, float], None, None]:
        if False:
            return 10
        return self._find_match(text, skip_lines, history, previous=False)

    def _find_previous_match(self, text: str, skip_lines: float, history: History):
        if False:
            while True:
                i = 10
        return reversed(list(self._find_match(text, skip_lines, history, previous=True)))

    def up(self, query: str, other_than: str, history: History) -> None:
        if False:
            print('Hello World!')
        for (suggestion, line_number) in self._find_next_match(query, self.skip_lines, history):
            if query + suggestion != other_than:
                self.skip_lines = line_number
                break
        else:
            self.skip_lines = 0

    def down(self, query: str, other_than: str, history: History) -> None:
        if False:
            while True:
                i = 10
        for (suggestion, line_number) in self._find_previous_match(query, self.skip_lines, history):
            if query + suggestion != other_than:
                self.skip_lines = line_number
                break
        else:
            for (suggestion, line_number) in self._find_previous_match(query, float('Inf'), history):
                if query + suggestion != other_than:
                    self.skip_lines = line_number
                    break

def accept_or_jump_to_end(event: KeyPressEvent):
    if False:
        i = 10
        return i + 15
    'Apply autosuggestion or jump to end of line.'
    buffer = event.current_buffer
    d = buffer.document
    after_cursor = d.text[d.cursor_position:]
    lines = after_cursor.split('\n')
    end_of_current_line = lines[0].strip()
    suggestion = buffer.suggestion
    if suggestion is not None and suggestion.text and (end_of_current_line == ''):
        buffer.insert_text(suggestion.text)
    else:
        nc.end_of_line(event)

def _deprected_accept_in_vi_insert_mode(event: KeyPressEvent):
    if False:
        while True:
            i = 10
    'Accept autosuggestion or jump to end of line.\n\n    .. deprecated:: 8.12\n        Use `accept_or_jump_to_end` instead.\n    '
    return accept_or_jump_to_end(event)

def accept(event: KeyPressEvent):
    if False:
        return 10
    'Accept autosuggestion'
    buffer = event.current_buffer
    suggestion = buffer.suggestion
    if suggestion:
        buffer.insert_text(suggestion.text)
    else:
        nc.forward_char(event)

def discard(event: KeyPressEvent):
    if False:
        for i in range(10):
            print('nop')
    'Discard autosuggestion'
    buffer = event.current_buffer
    buffer.suggestion = None

def accept_word(event: KeyPressEvent):
    if False:
        return 10
    'Fill partial autosuggestion by word'
    buffer = event.current_buffer
    suggestion = buffer.suggestion
    if suggestion:
        t = re.split('(\\S+\\s+)', suggestion.text)
        buffer.insert_text(next((x for x in t if x), ''))
    else:
        nc.forward_word(event)

def accept_character(event: KeyPressEvent):
    if False:
        i = 10
        return i + 15
    'Fill partial autosuggestion by character'
    b = event.current_buffer
    suggestion = b.suggestion
    if suggestion and suggestion.text:
        b.insert_text(suggestion.text[0])

def accept_and_keep_cursor(event: KeyPressEvent):
    if False:
        while True:
            i = 10
    'Accept autosuggestion and keep cursor in place'
    buffer = event.current_buffer
    old_position = buffer.cursor_position
    suggestion = buffer.suggestion
    if suggestion:
        buffer.insert_text(suggestion.text)
        buffer.cursor_position = old_position

def accept_and_move_cursor_left(event: KeyPressEvent):
    if False:
        return 10
    'Accept autosuggestion and move cursor left in place'
    accept_and_keep_cursor(event)
    nc.backward_char(event)

def _update_hint(buffer: Buffer):
    if False:
        print('Hello World!')
    if buffer.auto_suggest:
        suggestion = buffer.auto_suggest.get_suggestion(buffer, buffer.document)
        buffer.suggestion = suggestion

def backspace_and_resume_hint(event: KeyPressEvent):
    if False:
        print('Hello World!')
    'Resume autosuggestions after deleting last character'
    nc.backward_delete_char(event)
    _update_hint(event.current_buffer)

def resume_hinting(event: KeyPressEvent):
    if False:
        for i in range(10):
            print('nop')
    'Resume autosuggestions'
    pass_through.reply(event)
    _update_hint(event.current_buffer)

def up_and_update_hint(event: KeyPressEvent):
    if False:
        print('Hello World!')
    'Go up and update hint'
    current_buffer = event.current_buffer
    current_buffer.auto_up(count=event.arg)
    _update_hint(current_buffer)

def down_and_update_hint(event: KeyPressEvent):
    if False:
        for i in range(10):
            print('nop')
    'Go down and update hint'
    current_buffer = event.current_buffer
    current_buffer.auto_down(count=event.arg)
    _update_hint(current_buffer)

def accept_token(event: KeyPressEvent):
    if False:
        for i in range(10):
            print('nop')
    'Fill partial autosuggestion by token'
    b = event.current_buffer
    suggestion = b.suggestion
    if suggestion:
        prefix = _get_query(b.document)
        text = prefix + suggestion.text
        tokens: List[Optional[str]] = [None, None, None]
        substrings = ['']
        i = 0
        for token in generate_tokens(StringIO(text).readline):
            if token.type == tokenize.NEWLINE:
                index = len(text)
            else:
                index = text.index(token[1], len(substrings[-1]))
            substrings.append(text[:index])
            tokenized_so_far = substrings[-1]
            if tokenized_so_far.startswith(prefix):
                if i == 0 and len(tokenized_so_far) > len(prefix):
                    tokens[0] = tokenized_so_far[len(prefix):]
                    substrings.append(tokenized_so_far)
                    i += 1
                tokens[i] = token[1]
                if i == 2:
                    break
                i += 1
        if tokens[0]:
            to_insert: str
            insert_text = substrings[-2]
            if tokens[1] and len(tokens[1]) == 1:
                insert_text = substrings[-1]
            to_insert = insert_text[len(prefix):]
            b.insert_text(to_insert)
            return
    nc.forward_word(event)
Provider = Union[AutoSuggestFromHistory, NavigableAutoSuggestFromHistory, None]

def _swap_autosuggestion(buffer: Buffer, provider: NavigableAutoSuggestFromHistory, direction_method: Callable):
    if False:
        while True:
            i = 10
    '\n    We skip most recent history entry (in either direction) if it equals the\n    current autosuggestion because if user cycles when auto-suggestion is shown\n    they most likely want something else than what was suggested (otherwise\n    they would have accepted the suggestion).\n    '
    suggestion = buffer.suggestion
    if not suggestion:
        return
    query = _get_query(buffer.document)
    current = query + suggestion.text
    direction_method(query=query, other_than=current, history=buffer.history)
    new_suggestion = provider.get_suggestion(buffer, buffer.document)
    buffer.suggestion = new_suggestion

def swap_autosuggestion_up(event: KeyPressEvent):
    if False:
        i = 10
        return i + 15
    'Get next autosuggestion from history.'
    shell = get_ipython()
    provider = shell.auto_suggest
    if not isinstance(provider, NavigableAutoSuggestFromHistory):
        return
    return _swap_autosuggestion(buffer=event.current_buffer, provider=provider, direction_method=provider.up)

def swap_autosuggestion_down(event: KeyPressEvent):
    if False:
        while True:
            i = 10
    'Get previous autosuggestion from history.'
    shell = get_ipython()
    provider = shell.auto_suggest
    if not isinstance(provider, NavigableAutoSuggestFromHistory):
        return
    return _swap_autosuggestion(buffer=event.current_buffer, provider=provider, direction_method=provider.down)

def __getattr__(key):
    if False:
        print('Hello World!')
    if key == 'accept_in_vi_insert_mode':
        warnings.warn('`accept_in_vi_insert_mode` is deprecated since IPython 8.12 and renamed to `accept_or_jump_to_end`. Please update your configuration accordingly', DeprecationWarning, stacklevel=2)
        return _deprected_accept_in_vi_insert_mode
    raise AttributeError