"""
Handlers for Content-Encoding.

See: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Encoding
"""
import codecs
import io
import typing
import zlib
from ._compat import brotli
from ._exceptions import DecodingError

class ContentDecoder:

    def decode(self, data: bytes) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def flush(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

class IdentityDecoder(ContentDecoder):
    """
    Handle unencoded data.
    """

    def decode(self, data: bytes) -> bytes:
        if False:
            while True:
                i = 10
        return data

    def flush(self) -> bytes:
        if False:
            print('Hello World!')
        return b''

class DeflateDecoder(ContentDecoder):
    """
    Handle 'deflate' decoding.

    See: https://stackoverflow.com/questions/1838699
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.first_attempt = True
        self.decompressor = zlib.decompressobj()

    def decode(self, data: bytes) -> bytes:
        if False:
            return 10
        was_first_attempt = self.first_attempt
        self.first_attempt = False
        try:
            return self.decompressor.decompress(data)
        except zlib.error as exc:
            if was_first_attempt:
                self.decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
                return self.decode(data)
            raise DecodingError(str(exc)) from exc

    def flush(self) -> bytes:
        if False:
            print('Hello World!')
        try:
            return self.decompressor.flush()
        except zlib.error as exc:
            raise DecodingError(str(exc)) from exc

class GZipDecoder(ContentDecoder):
    """
    Handle 'gzip' decoding.

    See: https://stackoverflow.com/questions/1838699
    """

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)

    def decode(self, data: bytes) -> bytes:
        if False:
            print('Hello World!')
        try:
            return self.decompressor.decompress(data)
        except zlib.error as exc:
            raise DecodingError(str(exc)) from exc

    def flush(self) -> bytes:
        if False:
            while True:
                i = 10
        try:
            return self.decompressor.flush()
        except zlib.error as exc:
            raise DecodingError(str(exc)) from exc

class BrotliDecoder(ContentDecoder):
    """
    Handle 'brotli' decoding.

    Requires `pip install brotlipy`. See: https://brotlipy.readthedocs.io/
        or   `pip install brotli`. See https://github.com/google/brotli
    Supports both 'brotlipy' and 'Brotli' packages since they share an import
    name. The top branches are for 'brotlipy' and bottom branches for 'Brotli'
    """

    def __init__(self) -> None:
        if False:
            return 10
        if brotli is None:
            raise ImportError("Using 'BrotliDecoder', but neither of the 'brotlicffi' or 'brotli' packages have been installed. Make sure to install httpx using `pip install httpx[brotli]`.") from None
        self.decompressor = brotli.Decompressor()
        self.seen_data = False
        self._decompress: typing.Callable[[bytes], bytes]
        if hasattr(self.decompressor, 'decompress'):
            self._decompress = self.decompressor.decompress
        else:
            self._decompress = self.decompressor.process

    def decode(self, data: bytes) -> bytes:
        if False:
            while True:
                i = 10
        if not data:
            return b''
        self.seen_data = True
        try:
            return self._decompress(data)
        except brotli.error as exc:
            raise DecodingError(str(exc)) from exc

    def flush(self) -> bytes:
        if False:
            print('Hello World!')
        if not self.seen_data:
            return b''
        try:
            if hasattr(self.decompressor, 'finish'):
                self.decompressor.finish()
            return b''
        except brotli.error as exc:
            raise DecodingError(str(exc)) from exc

class MultiDecoder(ContentDecoder):
    """
    Handle the case where multiple encodings have been applied.
    """

    def __init__(self, children: typing.Sequence[ContentDecoder]) -> None:
        if False:
            return 10
        "\n        'children' should be a sequence of decoders in the order in which\n        each was applied.\n        "
        self.children = list(reversed(children))

    def decode(self, data: bytes) -> bytes:
        if False:
            print('Hello World!')
        for child in self.children:
            data = child.decode(data)
        return data

    def flush(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        data = b''
        for child in self.children:
            data = child.decode(data) + child.flush()
        return data

class ByteChunker:
    """
    Handles returning byte content in fixed-size chunks.
    """

    def __init__(self, chunk_size: typing.Optional[int]=None) -> None:
        if False:
            i = 10
            return i + 15
        self._buffer = io.BytesIO()
        self._chunk_size = chunk_size

    def decode(self, content: bytes) -> typing.List[bytes]:
        if False:
            return 10
        if self._chunk_size is None:
            return [content] if content else []
        self._buffer.write(content)
        if self._buffer.tell() >= self._chunk_size:
            value = self._buffer.getvalue()
            chunks = [value[i:i + self._chunk_size] for i in range(0, len(value), self._chunk_size)]
            if len(chunks[-1]) == self._chunk_size:
                self._buffer.seek(0)
                self._buffer.truncate()
                return chunks
            else:
                self._buffer.seek(0)
                self._buffer.write(chunks[-1])
                self._buffer.truncate()
                return chunks[:-1]
        else:
            return []

    def flush(self) -> typing.List[bytes]:
        if False:
            while True:
                i = 10
        value = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate()
        return [value] if value else []

class TextChunker:
    """
    Handles returning text content in fixed-size chunks.
    """

    def __init__(self, chunk_size: typing.Optional[int]=None) -> None:
        if False:
            while True:
                i = 10
        self._buffer = io.StringIO()
        self._chunk_size = chunk_size

    def decode(self, content: str) -> typing.List[str]:
        if False:
            while True:
                i = 10
        if self._chunk_size is None:
            return [content]
        self._buffer.write(content)
        if self._buffer.tell() >= self._chunk_size:
            value = self._buffer.getvalue()
            chunks = [value[i:i + self._chunk_size] for i in range(0, len(value), self._chunk_size)]
            if len(chunks[-1]) == self._chunk_size:
                self._buffer.seek(0)
                self._buffer.truncate()
                return chunks
            else:
                self._buffer.seek(0)
                self._buffer.write(chunks[-1])
                self._buffer.truncate()
                return chunks[:-1]
        else:
            return []

    def flush(self) -> typing.List[str]:
        if False:
            i = 10
            return i + 15
        value = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate()
        return [value] if value else []

class TextDecoder:
    """
    Handles incrementally decoding bytes into text
    """

    def __init__(self, encoding: str='utf-8'):
        if False:
            return 10
        self.decoder = codecs.getincrementaldecoder(encoding)(errors='replace')

    def decode(self, data: bytes) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.decoder.decode(data)

    def flush(self) -> str:
        if False:
            return 10
        return self.decoder.decode(b'', True)

class LineDecoder:
    """
    Handles incrementally reading lines from text.

    Has the same behaviour as the stdllib splitlines, but handling the input iteratively.
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.buffer: typing.List[str] = []
        self.trailing_cr: bool = False

    def decode(self, text: str) -> typing.List[str]:
        if False:
            print('Hello World!')
        NEWLINE_CHARS = '\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029'
        if self.trailing_cr:
            text = '\r' + text
            self.trailing_cr = False
        if text.endswith('\r'):
            self.trailing_cr = True
            text = text[:-1]
        if not text:
            return []
        trailing_newline = text[-1] in NEWLINE_CHARS
        lines = text.splitlines()
        if len(lines) == 1 and (not trailing_newline):
            self.buffer.append(lines[0])
            return []
        if self.buffer:
            lines = [''.join(self.buffer) + lines[0]] + lines[1:]
            self.buffer = []
        if not trailing_newline:
            self.buffer = [lines.pop()]
        return lines

    def flush(self) -> typing.List[str]:
        if False:
            print('Hello World!')
        if not self.buffer and (not self.trailing_cr):
            return []
        lines = [''.join(self.buffer)]
        self.buffer = []
        self.trailing_cr = False
        return lines
SUPPORTED_DECODERS = {'identity': IdentityDecoder, 'gzip': GZipDecoder, 'deflate': DeflateDecoder, 'br': BrotliDecoder}
if brotli is None:
    SUPPORTED_DECODERS.pop('br')