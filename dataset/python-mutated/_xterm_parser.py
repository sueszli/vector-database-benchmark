from __future__ import annotations
import re
import unicodedata
from typing import Any, Callable, Generator, Iterable
from . import events, messages
from ._ansi_sequences import ANSI_SEQUENCES_KEYS
from ._parser import Awaitable, Parser, TokenCallback
from .keys import KEY_NAME_REPLACEMENTS, _character_to_key
_MAX_SEQUENCE_SEARCH_THRESHOLD = 20
_re_mouse_event = re.compile('^' + re.escape('\x1b[') + '(<?[\\d;]+[mM]|M...)\\Z')
_re_terminal_mode_response = re.compile('^' + re.escape('\x1b[') + '\\?(?P<mode_id>\\d+);(?P<setting_parameter>\\d)\\$y')
_re_bracketed_paste_start = re.compile('^\\x1b\\[200~$')
_re_bracketed_paste_end = re.compile('^\\x1b\\[201~$')

class XTermParser(Parser[events.Event]):
    _re_sgr_mouse = re.compile('\\x1b\\[<(\\d+);(\\d+);(\\d+)([Mm])')

    def __init__(self, more_data: Callable[[], bool], debug: bool=False) -> None:
        if False:
            print('Hello World!')
        self.more_data = more_data
        self.last_x = 0
        self.last_y = 0
        self._debug_log_file = open('keys.log', 'wt') if debug else None
        super().__init__()

    def debug_log(self, *args: Any) -> None:
        if False:
            while True:
                i = 10
        if self._debug_log_file is not None:
            self._debug_log_file.write(' '.join(args) + '\n')
            self._debug_log_file.flush()

    def feed(self, data: str) -> Iterable[events.Event]:
        if False:
            for i in range(10):
                print('nop')
        self.debug_log(f'FEED {data!r}')
        return super().feed(data)

    def parse_mouse_code(self, code: str) -> events.Event | None:
        if False:
            while True:
                i = 10
        sgr_match = self._re_sgr_mouse.match(code)
        if sgr_match:
            (_buttons, _x, _y, state) = sgr_match.groups()
            buttons = int(_buttons)
            x = int(_x) - 1
            y = int(_y) - 1
            delta_x = x - self.last_x
            delta_y = y - self.last_y
            self.last_x = x
            self.last_y = y
            event_class: type[events.MouseEvent]
            if buttons & 64:
                event_class = events.MouseScrollDown if buttons & 1 else events.MouseScrollUp
                button = 0
            else:
                if buttons & 32:
                    event_class = events.MouseMove
                else:
                    event_class = events.MouseDown if state == 'M' else events.MouseUp
                button = buttons + 1 & 3
            event = event_class(x, y, delta_x, delta_y, button, bool(buttons & 4), bool(buttons & 8), bool(buttons & 16), screen_x=x, screen_y=y)
            return event
        return None
    _reissued_sequence_debug_book: Callable[[str], None] | None = None
    'INTERNAL USE ONLY!\n\n    If this property is set to a callable, it will be called *instead* of\n    the reissued sequence being emitted as key events.\n    '

    def parse(self, on_token: TokenCallback) -> Generator[Awaitable, str, None]:
        if False:
            print('Hello World!')
        ESC = '\x1b'
        read1 = self.read1
        sequence_to_key_events = self._sequence_to_key_events
        more_data = self.more_data
        paste_buffer: list[str] = []
        bracketed_paste = False
        use_prior_escape = False

        def reissue_sequence_as_keys(reissue_sequence: str) -> None:
            if False:
                while True:
                    i = 10
            if self._reissued_sequence_debug_book is not None:
                self._reissued_sequence_debug_book(reissue_sequence)
                return
            for character in reissue_sequence:
                key_events = sequence_to_key_events(character)
                for event in key_events:
                    if event.key == 'escape':
                        event = events.Key('circumflex_accent', '^')
                    on_token(event)
        while not self.is_eof:
            if not bracketed_paste and paste_buffer:
                pasted_text = ''.join(paste_buffer[:-1])
                on_token(events.Paste(pasted_text.replace('\x00', '')))
                paste_buffer.clear()
            character = ESC if use_prior_escape else (yield read1())
            use_prior_escape = False
            if bracketed_paste:
                paste_buffer.append(character)
            self.debug_log(f'character={character!r}')
            if character == ESC:
                sequence: str = character
                if not bracketed_paste:
                    peek_buffer = (yield self.peek_buffer())
                    if not peek_buffer:
                        on_token(events.Key('escape', '\x1b'))
                        continue
                    if peek_buffer and peek_buffer[0] == ESC:
                        yield read1()
                        on_token(events.Key('escape', '\x1b'))
                        if len(peek_buffer) == 1 and (not more_data()):
                            continue
                while True:
                    sequence_character = (yield read1())
                    new_sequence = sequence + sequence_character
                    threshold_exceeded = len(sequence) > _MAX_SEQUENCE_SEARCH_THRESHOLD
                    found_escape = sequence_character and sequence_character == ESC
                    if threshold_exceeded:
                        reissue_sequence_as_keys(new_sequence)
                        break
                    if found_escape:
                        use_prior_escape = True
                        reissue_sequence_as_keys(sequence)
                        break
                    sequence = new_sequence
                    self.debug_log(f'sequence={sequence!r}')
                    bracketed_paste_start_match = _re_bracketed_paste_start.match(sequence)
                    if bracketed_paste_start_match is not None:
                        bracketed_paste = True
                        break
                    bracketed_paste_end_match = _re_bracketed_paste_end.match(sequence)
                    if bracketed_paste_end_match is not None:
                        bracketed_paste = False
                        break
                    if not bracketed_paste:
                        key_events = list(sequence_to_key_events(sequence))
                        for key_event in key_events:
                            on_token(key_event)
                        if key_events:
                            break
                        mouse_match = _re_mouse_event.match(sequence)
                        if mouse_match is not None:
                            mouse_code = mouse_match.group(0)
                            event = self.parse_mouse_code(mouse_code)
                            if event:
                                on_token(event)
                            break
                        mode_report_match = _re_terminal_mode_response.match(sequence)
                        if mode_report_match is not None:
                            if mode_report_match['mode_id'] == '2026' and int(mode_report_match['setting_parameter']) > 0:
                                on_token(messages.TerminalSupportsSynchronizedOutput())
                            break
            elif not bracketed_paste:
                for event in sequence_to_key_events(character):
                    on_token(event)

    def _sequence_to_key_events(self, sequence: str, _unicode_name=unicodedata.name) -> Iterable[events.Key]:
        if False:
            while True:
                i = 10
        'Map a sequence of code points on to a sequence of keys.\n\n        Args:\n            sequence: Sequence of code points.\n\n        Returns:\n            Keys\n        '
        keys = ANSI_SEQUENCES_KEYS.get(sequence)
        if keys is not None:
            for key in keys:
                yield events.Key(key.value, sequence if len(sequence) == 1 else None)
        elif len(sequence) == 1:
            try:
                if not sequence.isalnum():
                    name = _character_to_key(sequence)
                else:
                    name = sequence
                name = KEY_NAME_REPLACEMENTS.get(name, name)
                yield events.Key(name, sequence)
            except:
                yield events.Key(sequence, sequence)