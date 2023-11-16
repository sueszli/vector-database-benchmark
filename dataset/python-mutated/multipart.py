from __future__ import annotations
import re
import typing as t
from dataclasses import dataclass
from enum import auto
from enum import Enum
from ..datastructures import Headers
from ..exceptions import RequestEntityTooLarge
from ..http import parse_options_header

class Event:
    pass

@dataclass(frozen=True)
class Preamble(Event):
    data: bytes

@dataclass(frozen=True)
class Field(Event):
    name: str
    headers: Headers

@dataclass(frozen=True)
class File(Event):
    name: str
    filename: str
    headers: Headers

@dataclass(frozen=True)
class Data(Event):
    data: bytes
    more_data: bool

@dataclass(frozen=True)
class Epilogue(Event):
    data: bytes

class NeedData(Event):
    pass
NEED_DATA = NeedData()

class State(Enum):
    PREAMBLE = auto()
    PART = auto()
    DATA = auto()
    DATA_START = auto()
    EPILOGUE = auto()
    COMPLETE = auto()
LINE_BREAK = b'(?:\r\n|\n|\r)'
BLANK_LINE_RE = re.compile(b'(?:\r\n\r\n|\r\r|\n\n)', re.MULTILINE)
LINE_BREAK_RE = re.compile(LINE_BREAK, re.MULTILINE)
HEADER_CONTINUATION_RE = re.compile(b'%s[ \t]' % LINE_BREAK, re.MULTILINE)
SEARCH_EXTRA_LENGTH = 8

class MultipartDecoder:
    """Decodes a multipart message as bytes into Python events.

    The part data is returned as available to allow the caller to save
    the data from memory to disk, if desired.
    """

    def __init__(self, boundary: bytes, max_form_memory_size: int | None=None, *, max_parts: int | None=None) -> None:
        if False:
            return 10
        self.buffer = bytearray()
        self.complete = False
        self.max_form_memory_size = max_form_memory_size
        self.max_parts = max_parts
        self.state = State.PREAMBLE
        self.boundary = boundary
        self.preamble_re = re.compile(b'%s?--%s(--[^\\S\\n\\r]*%s?|[^\\S\\n\\r]*%s)' % (LINE_BREAK, re.escape(boundary), LINE_BREAK, LINE_BREAK), re.MULTILINE)
        self.boundary_re = re.compile(b'%s--%s(--[^\\S\\n\\r]*%s?|[^\\S\\n\\r]*%s)' % (LINE_BREAK, re.escape(boundary), LINE_BREAK, LINE_BREAK), re.MULTILINE)
        self._search_position = 0
        self._parts_decoded = 0

    def last_newline(self, data: bytes) -> int:
        if False:
            return 10
        try:
            last_nl = data.rindex(b'\n')
        except ValueError:
            last_nl = len(data)
        try:
            last_cr = data.rindex(b'\r')
        except ValueError:
            last_cr = len(data)
        return min(last_nl, last_cr)

    def receive_data(self, data: bytes | None) -> None:
        if False:
            return 10
        if data is None:
            self.complete = True
        elif self.max_form_memory_size is not None and len(self.buffer) + len(data) > self.max_form_memory_size:
            raise RequestEntityTooLarge()
        else:
            self.buffer.extend(data)

    def next_event(self) -> Event:
        if False:
            print('Hello World!')
        event: Event = NEED_DATA
        if self.state == State.PREAMBLE:
            match = self.preamble_re.search(self.buffer, self._search_position)
            if match is not None:
                if match.group(1).startswith(b'--'):
                    self.state = State.EPILOGUE
                else:
                    self.state = State.PART
                data = bytes(self.buffer[:match.start()])
                del self.buffer[:match.end()]
                event = Preamble(data=data)
                self._search_position = 0
            else:
                self._search_position = max(0, len(self.buffer) - len(self.boundary) - SEARCH_EXTRA_LENGTH)
        elif self.state == State.PART:
            match = BLANK_LINE_RE.search(self.buffer, self._search_position)
            if match is not None:
                headers = self._parse_headers(self.buffer[:match.start()])
                headers_end = (match.start() + match.end()) // 2
                del self.buffer[:headers_end]
                if 'content-disposition' not in headers:
                    raise ValueError('Missing Content-Disposition header')
                (disposition, extra) = parse_options_header(headers['content-disposition'])
                name = t.cast(str, extra.get('name'))
                filename = extra.get('filename')
                if filename is not None:
                    event = File(filename=filename, headers=headers, name=name)
                else:
                    event = Field(headers=headers, name=name)
                self.state = State.DATA_START
                self._search_position = 0
                self._parts_decoded += 1
                if self.max_parts is not None and self._parts_decoded > self.max_parts:
                    raise RequestEntityTooLarge()
            else:
                self._search_position = max(0, len(self.buffer) - SEARCH_EXTRA_LENGTH)
        elif self.state == State.DATA_START:
            (data, del_index, more_data) = self._parse_data(self.buffer, start=True)
            del self.buffer[:del_index]
            event = Data(data=data, more_data=more_data)
            if more_data:
                self.state = State.DATA
        elif self.state == State.DATA:
            (data, del_index, more_data) = self._parse_data(self.buffer, start=False)
            del self.buffer[:del_index]
            if data or not more_data:
                event = Data(data=data, more_data=more_data)
        elif self.state == State.EPILOGUE and self.complete:
            event = Epilogue(data=bytes(self.buffer))
            del self.buffer[:]
            self.state = State.COMPLETE
        if self.complete and isinstance(event, NeedData):
            raise ValueError(f'Invalid form-data cannot parse beyond {self.state}')
        return event

    def _parse_headers(self, data: bytes) -> Headers:
        if False:
            return 10
        headers: list[tuple[str, str]] = []
        data = HEADER_CONTINUATION_RE.sub(b' ', data)
        for line in data.splitlines():
            line = line.strip()
            if line != b'':
                (name, _, value) = line.decode().partition(':')
                headers.append((name.strip(), value.strip()))
        return Headers(headers)

    def _parse_data(self, data: bytes, *, start: bool) -> tuple[bytes, int, bool]:
        if False:
            while True:
                i = 10
        if start:
            match = LINE_BREAK_RE.match(data)
            data_start = t.cast(t.Match[bytes], match).end()
        else:
            data_start = 0
        boundary = b'--' + self.boundary
        if self.buffer.find(boundary) == -1:
            data_end = del_index = self.last_newline(data[data_start:]) + data_start
            if len(data) - data_end > len(b'\n' + boundary):
                data_end = del_index = len(data)
            more_data = True
        else:
            match = self.boundary_re.search(data)
            if match is not None:
                if match.group(1).startswith(b'--'):
                    self.state = State.EPILOGUE
                else:
                    self.state = State.PART
                data_end = match.start()
                del_index = match.end()
            else:
                data_end = del_index = self.last_newline(data[data_start:]) + data_start
            more_data = match is None
        return (bytes(data[data_start:data_end]), del_index, more_data)

class MultipartEncoder:

    def __init__(self, boundary: bytes) -> None:
        if False:
            i = 10
            return i + 15
        self.boundary = boundary
        self.state = State.PREAMBLE

    def send_event(self, event: Event) -> bytes:
        if False:
            i = 10
            return i + 15
        if isinstance(event, Preamble) and self.state == State.PREAMBLE:
            self.state = State.PART
            return event.data
        elif isinstance(event, (Field, File)) and self.state in {State.PREAMBLE, State.PART, State.DATA}:
            data = b'\r\n--' + self.boundary + b'\r\n'
            data += b'Content-Disposition: form-data; name="%s"' % event.name.encode()
            if isinstance(event, File):
                data += b'; filename="%s"' % event.filename.encode()
            data += b'\r\n'
            for (name, value) in t.cast(Field, event).headers:
                if name.lower() != 'content-disposition':
                    data += f'{name}: {value}\r\n'.encode()
            self.state = State.DATA_START
            return data
        elif isinstance(event, Data) and self.state == State.DATA_START:
            self.state = State.DATA
            if len(event.data) > 0:
                return b'\r\n' + event.data
            else:
                return event.data
        elif isinstance(event, Data) and self.state == State.DATA:
            return event.data
        elif isinstance(event, Epilogue):
            self.state = State.COMPLETE
            return b'\r\n--' + self.boundary + b'--\r\n' + event.data
        else:
            raise ValueError(f'Cannot generate {event} in state: {self.state}')