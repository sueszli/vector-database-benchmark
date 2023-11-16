import typing
from dataclasses import dataclass, field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus
from starlette.datastructures import FormData, Headers, UploadFile
try:
    import multipart
    from multipart.multipart import parse_options_header
except ModuleNotFoundError:
    parse_options_header = None
    multipart = None

class FormMessage(Enum):
    FIELD_START = 1
    FIELD_NAME = 2
    FIELD_DATA = 3
    FIELD_END = 4
    END = 5

@dataclass
class MultipartPart:
    content_disposition: typing.Optional[bytes] = None
    field_name: str = ''
    data: bytes = b''
    file: typing.Optional[UploadFile] = None
    item_headers: typing.List[typing.Tuple[bytes, bytes]] = field(default_factory=list)

def _user_safe_decode(src: bytes, codec: str) -> str:
    if False:
        return 10
    try:
        return src.decode(codec)
    except (UnicodeDecodeError, LookupError):
        return src.decode('latin-1')

class MultiPartException(Exception):

    def __init__(self, message: str) -> None:
        if False:
            while True:
                i = 10
        self.message = message

class FormParser:

    def __init__(self, headers: Headers, stream: typing.AsyncGenerator[bytes, None]) -> None:
        if False:
            return 10
        assert multipart is not None, 'The `python-multipart` library must be installed to use form parsing.'
        self.headers = headers
        self.stream = stream
        self.messages: typing.List[typing.Tuple[FormMessage, bytes]] = []

    def on_field_start(self) -> None:
        if False:
            i = 10
            return i + 15
        message = (FormMessage.FIELD_START, b'')
        self.messages.append(message)

    def on_field_name(self, data: bytes, start: int, end: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        message = (FormMessage.FIELD_NAME, data[start:end])
        self.messages.append(message)

    def on_field_data(self, data: bytes, start: int, end: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        message = (FormMessage.FIELD_DATA, data[start:end])
        self.messages.append(message)

    def on_field_end(self) -> None:
        if False:
            return 10
        message = (FormMessage.FIELD_END, b'')
        self.messages.append(message)

    def on_end(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        message = (FormMessage.END, b'')
        self.messages.append(message)

    async def parse(self) -> FormData:
        callbacks = {'on_field_start': self.on_field_start, 'on_field_name': self.on_field_name, 'on_field_data': self.on_field_data, 'on_field_end': self.on_field_end, 'on_end': self.on_end}
        parser = multipart.QuerystringParser(callbacks)
        field_name = b''
        field_value = b''
        items: typing.List[typing.Tuple[str, typing.Union[str, UploadFile]]] = []
        async for chunk in self.stream:
            if chunk:
                parser.write(chunk)
            else:
                parser.finalize()
            messages = list(self.messages)
            self.messages.clear()
            for (message_type, message_bytes) in messages:
                if message_type == FormMessage.FIELD_START:
                    field_name = b''
                    field_value = b''
                elif message_type == FormMessage.FIELD_NAME:
                    field_name += message_bytes
                elif message_type == FormMessage.FIELD_DATA:
                    field_value += message_bytes
                elif message_type == FormMessage.FIELD_END:
                    name = unquote_plus(field_name.decode('latin-1'))
                    value = unquote_plus(field_value.decode('latin-1'))
                    items.append((name, value))
        return FormData(items)

class MultiPartParser:
    max_file_size = 1024 * 1024

    def __init__(self, headers: Headers, stream: typing.AsyncGenerator[bytes, None], *, max_files: typing.Union[int, float]=1000, max_fields: typing.Union[int, float]=1000) -> None:
        if False:
            return 10
        assert multipart is not None, 'The `python-multipart` library must be installed to use form parsing.'
        self.headers = headers
        self.stream = stream
        self.max_files = max_files
        self.max_fields = max_fields
        self.items: typing.List[typing.Tuple[str, typing.Union[str, UploadFile]]] = []
        self._current_files = 0
        self._current_fields = 0
        self._current_partial_header_name: bytes = b''
        self._current_partial_header_value: bytes = b''
        self._current_part = MultipartPart()
        self._charset = ''
        self._file_parts_to_write: typing.List[typing.Tuple[MultipartPart, bytes]] = []
        self._file_parts_to_finish: typing.List[MultipartPart] = []
        self._files_to_close_on_error: typing.List[SpooledTemporaryFile[bytes]] = []

    def on_part_begin(self) -> None:
        if False:
            while True:
                i = 10
        self._current_part = MultipartPart()

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if False:
            print('Hello World!')
        message_bytes = data[start:end]
        if self._current_part.file is None:
            self._current_part.data += message_bytes
        else:
            self._file_parts_to_write.append((self._current_part, message_bytes))

    def on_part_end(self) -> None:
        if False:
            return 10
        if self._current_part.file is None:
            self.items.append((self._current_part.field_name, _user_safe_decode(self._current_part.data, self._charset)))
        else:
            self._file_parts_to_finish.append(self._current_part)
            self.items.append((self._current_part.field_name, self._current_part.file))

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._current_partial_header_name += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        if False:
            i = 10
            return i + 15
        self._current_partial_header_value += data[start:end]

    def on_header_end(self) -> None:
        if False:
            return 10
        field = self._current_partial_header_name.lower()
        if field == b'content-disposition':
            self._current_part.content_disposition = self._current_partial_header_value
        self._current_part.item_headers.append((field, self._current_partial_header_value))
        self._current_partial_header_name = b''
        self._current_partial_header_value = b''

    def on_headers_finished(self) -> None:
        if False:
            print('Hello World!')
        (disposition, options) = parse_options_header(self._current_part.content_disposition)
        try:
            self._current_part.field_name = _user_safe_decode(options[b'name'], self._charset)
        except KeyError:
            raise MultiPartException('The Content-Disposition header field "name" must be provided.')
        if b'filename' in options:
            self._current_files += 1
            if self._current_files > self.max_files:
                raise MultiPartException(f'Too many files. Maximum number of files is {self.max_files}.')
            filename = _user_safe_decode(options[b'filename'], self._charset)
            tempfile = SpooledTemporaryFile(max_size=self.max_file_size)
            self._files_to_close_on_error.append(tempfile)
            self._current_part.file = UploadFile(file=tempfile, size=0, filename=filename, headers=Headers(raw=self._current_part.item_headers))
        else:
            self._current_fields += 1
            if self._current_fields > self.max_fields:
                raise MultiPartException(f'Too many fields. Maximum number of fields is {self.max_fields}.')
            self._current_part.file = None

    def on_end(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    async def parse(self) -> FormData:
        (_, params) = parse_options_header(self.headers['Content-Type'])
        charset = params.get(b'charset', 'utf-8')
        if isinstance(charset, bytes):
            charset = charset.decode('latin-1')
        self._charset = charset
        try:
            boundary = params[b'boundary']
        except KeyError:
            raise MultiPartException('Missing boundary in multipart.')
        callbacks = {'on_part_begin': self.on_part_begin, 'on_part_data': self.on_part_data, 'on_part_end': self.on_part_end, 'on_header_field': self.on_header_field, 'on_header_value': self.on_header_value, 'on_header_end': self.on_header_end, 'on_headers_finished': self.on_headers_finished, 'on_end': self.on_end}
        parser = multipart.MultipartParser(boundary, callbacks)
        try:
            async for chunk in self.stream:
                parser.write(chunk)
                for (part, data) in self._file_parts_to_write:
                    assert part.file
                    await part.file.write(data)
                for part in self._file_parts_to_finish:
                    assert part.file
                    await part.file.seek(0)
                self._file_parts_to_write.clear()
                self._file_parts_to_finish.clear()
        except MultiPartException as exc:
            for file in self._files_to_close_on_error:
                file.close()
            raise exc
        parser.finalize()
        return FormData(self.items)