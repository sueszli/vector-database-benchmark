"""
Multi-part parsing for file uploads.

Exposes one class, ``MultiPartParser``, which feeds chunks of uploaded data to
file upload handlers for processing.
"""
import base64
import binascii
import collections
import html
from django.conf import settings
from django.core.exceptions import RequestDataTooBig, SuspiciousMultipartForm, TooManyFieldsSent, TooManyFilesSent
from django.core.files.uploadhandler import SkipFile, StopFutureHandlers, StopUpload
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.http import parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
__all__ = ('MultiPartParser', 'MultiPartParserError', 'InputStreamExhausted')

class MultiPartParserError(Exception):
    pass

class InputStreamExhausted(Exception):
    """
    No more reads are allowed from this device.
    """
    pass
RAW = 'raw'
FILE = 'file'
FIELD = 'field'
FIELD_TYPES = frozenset([FIELD, RAW])

class MultiPartParser:
    """
    An RFC 7578 multipart/form-data parser.

    ``MultiValueDict.parse()`` reads the input stream in ``chunk_size`` chunks
    and returns a tuple of ``(MultiValueDict(POST), MultiValueDict(FILES))``.
    """
    boundary_re = _lazy_re_compile('[ -~]{0,200}[!-~]')

    def __init__(self, META, input_data, upload_handlers, encoding=None):
        if False:
            return 10
        '\n        Initialize the MultiPartParser object.\n\n        :META:\n            The standard ``META`` dictionary in Django request objects.\n        :input_data:\n            The raw post data, as a file-like object.\n        :upload_handlers:\n            A list of UploadHandler instances that perform operations on the\n            uploaded data.\n        :encoding:\n            The encoding with which to treat the incoming data.\n        '
        content_type = META.get('CONTENT_TYPE', '')
        if not content_type.startswith('multipart/'):
            raise MultiPartParserError('Invalid Content-Type: %s' % content_type)
        try:
            content_type.encode('ascii')
        except UnicodeEncodeError:
            raise MultiPartParserError('Invalid non-ASCII Content-Type in multipart: %s' % force_str(content_type))
        (_, opts) = parse_header_parameters(content_type)
        boundary = opts.get('boundary')
        if not boundary or not self.boundary_re.fullmatch(boundary):
            raise MultiPartParserError('Invalid boundary in multipart: %s' % force_str(boundary))
        try:
            content_length = int(META.get('CONTENT_LENGTH', 0))
        except (ValueError, TypeError):
            content_length = 0
        if content_length < 0:
            raise MultiPartParserError('Invalid content length: %r' % content_length)
        self._boundary = boundary.encode('ascii')
        self._input_data = input_data
        possible_sizes = [x.chunk_size for x in upload_handlers if x.chunk_size]
        self._chunk_size = min([2 ** 31 - 4] + possible_sizes)
        self._meta = META
        self._encoding = encoding or settings.DEFAULT_CHARSET
        self._content_length = content_length
        self._upload_handlers = upload_handlers

    def parse(self):
        if False:
            print('Hello World!')
        try:
            return self._parse()
        except Exception:
            if hasattr(self, '_files'):
                for (_, files) in self._files.lists():
                    for fileobj in files:
                        fileobj.close()
            raise

    def _parse(self):
        if False:
            i = 10
            return i + 15
        '\n        Parse the POST data and break it into a FILES MultiValueDict and a POST\n        MultiValueDict.\n\n        Return a tuple containing the POST and FILES dictionary, respectively.\n        '
        from django.http import QueryDict
        encoding = self._encoding
        handlers = self._upload_handlers
        if self._content_length == 0:
            return (QueryDict(encoding=self._encoding), MultiValueDict())
        for handler in handlers:
            result = handler.handle_raw_input(self._input_data, self._meta, self._content_length, self._boundary, encoding)
            if result is not None:
                return (result[0], result[1])
        self._post = QueryDict(mutable=True)
        self._files = MultiValueDict()
        stream = LazyStream(ChunkIter(self._input_data, self._chunk_size))
        old_field_name = None
        counters = [0] * len(handlers)
        num_bytes_read = 0
        num_post_keys = 0
        num_files = 0
        read_size = None
        uploaded_file = True
        try:
            for (item_type, meta_data, field_stream) in Parser(stream, self._boundary):
                if old_field_name:
                    self.handle_file_complete(old_field_name, counters)
                    old_field_name = None
                    uploaded_file = True
                if item_type in FIELD_TYPES and settings.DATA_UPLOAD_MAX_NUMBER_FIELDS is not None:
                    num_post_keys += 1
                    if settings.DATA_UPLOAD_MAX_NUMBER_FIELDS + 2 < num_post_keys:
                        raise TooManyFieldsSent('The number of GET/POST parameters exceeded settings.DATA_UPLOAD_MAX_NUMBER_FIELDS.')
                try:
                    disposition = meta_data['content-disposition'][1]
                    field_name = disposition['name'].strip()
                except (KeyError, IndexError, AttributeError):
                    continue
                transfer_encoding = meta_data.get('content-transfer-encoding')
                if transfer_encoding is not None:
                    transfer_encoding = transfer_encoding[0].strip()
                field_name = force_str(field_name, encoding, errors='replace')
                if item_type == FIELD:
                    if settings.DATA_UPLOAD_MAX_MEMORY_SIZE is not None:
                        read_size = settings.DATA_UPLOAD_MAX_MEMORY_SIZE - num_bytes_read
                    if transfer_encoding == 'base64':
                        raw_data = field_stream.read(size=read_size)
                        num_bytes_read += len(raw_data)
                        try:
                            data = base64.b64decode(raw_data)
                        except binascii.Error:
                            data = raw_data
                    else:
                        data = field_stream.read(size=read_size)
                        num_bytes_read += len(data)
                    num_bytes_read += len(field_name) + 2
                    if settings.DATA_UPLOAD_MAX_MEMORY_SIZE is not None and num_bytes_read > settings.DATA_UPLOAD_MAX_MEMORY_SIZE:
                        raise RequestDataTooBig('Request body exceeded settings.DATA_UPLOAD_MAX_MEMORY_SIZE.')
                    self._post.appendlist(field_name, force_str(data, encoding, errors='replace'))
                elif item_type == FILE:
                    num_files += 1
                    if settings.DATA_UPLOAD_MAX_NUMBER_FILES is not None and num_files > settings.DATA_UPLOAD_MAX_NUMBER_FILES:
                        raise TooManyFilesSent('The number of files exceeded settings.DATA_UPLOAD_MAX_NUMBER_FILES.')
                    file_name = disposition.get('filename')
                    if file_name:
                        file_name = force_str(file_name, encoding, errors='replace')
                        file_name = self.sanitize_file_name(file_name)
                    if not file_name:
                        continue
                    (content_type, content_type_extra) = meta_data.get('content-type', ('', {}))
                    content_type = content_type.strip()
                    charset = content_type_extra.get('charset')
                    try:
                        content_length = int(meta_data.get('content-length')[0])
                    except (IndexError, TypeError, ValueError):
                        content_length = None
                    counters = [0] * len(handlers)
                    uploaded_file = False
                    try:
                        for handler in handlers:
                            try:
                                handler.new_file(field_name, file_name, content_type, content_length, charset, content_type_extra)
                            except StopFutureHandlers:
                                break
                        for chunk in field_stream:
                            if transfer_encoding == 'base64':
                                stripped_chunk = b''.join(chunk.split())
                                remaining = len(stripped_chunk) % 4
                                while remaining != 0:
                                    over_chunk = field_stream.read(4 - remaining)
                                    if not over_chunk:
                                        break
                                    stripped_chunk += b''.join(over_chunk.split())
                                    remaining = len(stripped_chunk) % 4
                                try:
                                    chunk = base64.b64decode(stripped_chunk)
                                except Exception as exc:
                                    raise MultiPartParserError('Could not decode base64 data.') from exc
                            for (i, handler) in enumerate(handlers):
                                chunk_length = len(chunk)
                                chunk = handler.receive_data_chunk(chunk, counters[i])
                                counters[i] += chunk_length
                                if chunk is None:
                                    break
                    except SkipFile:
                        self._close_files()
                        exhaust(field_stream)
                    else:
                        old_field_name = field_name
                else:
                    exhaust(field_stream)
        except StopUpload as e:
            self._close_files()
            if not e.connection_reset:
                exhaust(self._input_data)
        else:
            if not uploaded_file:
                for handler in handlers:
                    handler.upload_interrupted()
            exhaust(self._input_data)
        any((handler.upload_complete() for handler in handlers))
        self._post._mutable = False
        return (self._post, self._files)

    def handle_file_complete(self, old_field_name, counters):
        if False:
            i = 10
            return i + 15
        '\n        Handle all the signaling that takes place when a file is complete.\n        '
        for (i, handler) in enumerate(self._upload_handlers):
            file_obj = handler.file_complete(counters[i])
            if file_obj:
                self._files.appendlist(force_str(old_field_name, self._encoding, errors='replace'), file_obj)
                break

    def sanitize_file_name(self, file_name):
        if False:
            while True:
                i = 10
        '\n        Sanitize the filename of an upload.\n\n        Remove all possible path separators, even though that might remove more\n        than actually required by the target system. Filenames that could\n        potentially cause problems (current/parent dir) are also discarded.\n\n        It should be noted that this function could still return a "filepath"\n        like "C:some_file.txt" which is handled later on by the storage layer.\n        So while this function does sanitize filenames to some extent, the\n        resulting filename should still be considered as untrusted user input.\n        '
        file_name = html.unescape(file_name)
        file_name = file_name.rsplit('/')[-1]
        file_name = file_name.rsplit('\\')[-1]
        file_name = ''.join([char for char in file_name if char.isprintable()])
        if file_name in {'', '.', '..'}:
            return None
        return file_name
    IE_sanitize = sanitize_file_name

    def _close_files(self):
        if False:
            return 10
        for handler in self._upload_handlers:
            if hasattr(handler, 'file'):
                handler.file.close()

class LazyStream:
    """
    The LazyStream wrapper allows one to get and "unget" bytes from a stream.

    Given a producer object (an iterator that yields bytestrings), the
    LazyStream object will support iteration, reading, and keeping a "look-back"
    variable in case you need to "unget" some bytes.
    """

    def __init__(self, producer, length=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Every LazyStream must have a producer when instantiated.\n\n        A producer is an iterable that returns a string each time it\n        is called.\n        '
        self._producer = producer
        self._empty = False
        self._leftover = b''
        self.length = length
        self.position = 0
        self._remaining = length
        self._unget_history = []

    def tell(self):
        if False:
            print('Hello World!')
        return self.position

    def read(self, size=None):
        if False:
            return 10

        def parts():
            if False:
                while True:
                    i = 10
            remaining = self._remaining if size is None else size
            if remaining is None:
                yield b''.join(self)
                return
            while remaining != 0:
                assert remaining > 0, 'remaining bytes to read should never go negative'
                try:
                    chunk = next(self)
                except StopIteration:
                    return
                else:
                    emitting = chunk[:remaining]
                    self.unget(chunk[remaining:])
                    remaining -= len(emitting)
                    yield emitting
        return b''.join(parts())

    def __next__(self):
        if False:
            return 10
        '\n        Used when the exact number of bytes to read is unimportant.\n\n        Return whatever chunk is conveniently returned from the iterator.\n        Useful to avoid unnecessary bookkeeping if performance is an issue.\n        '
        if self._leftover:
            output = self._leftover
            self._leftover = b''
        else:
            output = next(self._producer)
            self._unget_history = []
        self.position += len(output)
        return output

    def close(self):
        if False:
            while True:
                i = 10
        '\n        Used to invalidate/disable this lazy stream.\n\n        Replace the producer with an empty list. Any leftover bytes that have\n        already been read will still be reported upon read() and/or next().\n        '
        self._producer = []

    def __iter__(self):
        if False:
            return 10
        return self

    def unget(self, bytes):
        if False:
            for i in range(10):
                print('nop')
        '\n        Place bytes back onto the front of the lazy stream.\n\n        Future calls to read() will return those bytes first. The\n        stream position and thus tell() will be rewound.\n        '
        if not bytes:
            return
        self._update_unget_history(len(bytes))
        self.position -= len(bytes)
        self._leftover = bytes + self._leftover

    def _update_unget_history(self, num_bytes):
        if False:
            i = 10
            return i + 15
        "\n        Update the unget history as a sanity check to see if we've pushed\n        back the same number of bytes in one chunk. If we keep ungetting the\n        same number of bytes many times (here, 50), we're mostly likely in an\n        infinite loop of some sort. This is usually caused by a\n        maliciously-malformed MIME request.\n        "
        self._unget_history = [num_bytes] + self._unget_history[:49]
        number_equal = len([current_number for current_number in self._unget_history if current_number == num_bytes])
        if number_equal > 40:
            raise SuspiciousMultipartForm("The multipart parser got stuck, which shouldn't happen with normal uploaded files. Check for malicious upload activity; if there is none, report this to the Django developers.")

class ChunkIter:
    """
    An iterable that will yield chunks of data. Given a file-like object as the
    constructor, yield chunks of read operations from that object.
    """

    def __init__(self, flo, chunk_size=64 * 1024):
        if False:
            while True:
                i = 10
        self.flo = flo
        self.chunk_size = chunk_size

    def __next__(self):
        if False:
            while True:
                i = 10
        try:
            data = self.flo.read(self.chunk_size)
        except InputStreamExhausted:
            raise StopIteration()
        if data:
            return data
        else:
            raise StopIteration()

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

class InterBoundaryIter:
    """
    A Producer that will iterate over boundaries.
    """

    def __init__(self, stream, boundary):
        if False:
            while True:
                i = 10
        self._stream = stream
        self._boundary = boundary

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        try:
            return LazyStream(BoundaryIter(self._stream, self._boundary))
        except InputStreamExhausted:
            raise StopIteration()

class BoundaryIter:
    """
    A Producer that is sensitive to boundaries.

    Will happily yield bytes until a boundary is found. Will yield the bytes
    before the boundary, throw away the boundary bytes themselves, and push the
    post-boundary bytes back on the stream.

    The future calls to next() after locating the boundary will raise a
    StopIteration exception.
    """

    def __init__(self, stream, boundary):
        if False:
            while True:
                i = 10
        self._stream = stream
        self._boundary = boundary
        self._done = False
        self._rollback = len(boundary) + 6
        unused_char = self._stream.read(1)
        if not unused_char:
            raise InputStreamExhausted()
        self._stream.unget(unused_char)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        if self._done:
            raise StopIteration()
        stream = self._stream
        rollback = self._rollback
        bytes_read = 0
        chunks = []
        for bytes in stream:
            bytes_read += len(bytes)
            chunks.append(bytes)
            if bytes_read > rollback:
                break
            if not bytes:
                break
        else:
            self._done = True
        if not chunks:
            raise StopIteration()
        chunk = b''.join(chunks)
        boundary = self._find_boundary(chunk)
        if boundary:
            (end, next) = boundary
            stream.unget(chunk[next:])
            self._done = True
            return chunk[:end]
        elif not chunk[:-rollback]:
            self._done = True
            return chunk
        else:
            stream.unget(chunk[-rollback:])
            return chunk[:-rollback]

    def _find_boundary(self, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find a multipart boundary in data.\n\n        Should no boundary exist in the data, return None. Otherwise, return\n        a tuple containing the indices of the following:\n         * the end of current encapsulation\n         * the start of the next encapsulation\n        '
        index = data.find(self._boundary)
        if index < 0:
            return None
        else:
            end = index
            next = index + len(self._boundary)
            last = max(0, end - 1)
            if data[last:last + 1] == b'\n':
                end -= 1
            last = max(0, end - 1)
            if data[last:last + 1] == b'\r':
                end -= 1
            return (end, next)

def exhaust(stream_or_iterable):
    if False:
        i = 10
        return i + 15
    'Exhaust an iterator or stream.'
    try:
        iterator = iter(stream_or_iterable)
    except TypeError:
        iterator = ChunkIter(stream_or_iterable, 16384)
    collections.deque(iterator, maxlen=0)

def parse_boundary_stream(stream, max_header_size):
    if False:
        print('Hello World!')
    '\n    Parse one and exactly one stream that encapsulates a boundary.\n    '
    chunk = stream.read(max_header_size)
    header_end = chunk.find(b'\r\n\r\n')
    if header_end == -1:
        stream.unget(chunk)
        return (RAW, {}, stream)
    header = chunk[:header_end]
    stream.unget(chunk[header_end + 4:])
    TYPE = RAW
    outdict = {}
    for line in header.split(b'\r\n'):
        try:
            (main_value_pair, params) = parse_header_parameters(line.decode())
            (name, value) = main_value_pair.split(':', 1)
            params = {k: v.encode() for (k, v) in params.items()}
        except ValueError:
            continue
        if name == 'content-disposition':
            TYPE = FIELD
            if params.get('filename'):
                TYPE = FILE
        outdict[name] = (value, params)
    if TYPE == RAW:
        stream.unget(chunk)
    return (TYPE, outdict, stream)

class Parser:

    def __init__(self, stream, boundary):
        if False:
            return 10
        self._stream = stream
        self._separator = b'--' + boundary

    def __iter__(self):
        if False:
            return 10
        boundarystream = InterBoundaryIter(self._stream, self._separator)
        for sub_stream in boundarystream:
            yield parse_boundary_stream(sub_stream, 1024)