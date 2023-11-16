import io
import struct
import zipfile
from typing import IO, List, Optional, Tuple, Union
from airbyte_cdk.sources.file_based.remote_file import RemoteFile
from botocore.client import BaseClient
from source_s3.v4.config import Config
BUFFER_SIZE_DEFAULT = 1024 * 1024
MAX_BUFFER_SIZE_DEFAULT: int = 16 * BUFFER_SIZE_DEFAULT

class RemoteFileInsideArchive(RemoteFile):
    """
    A file inside archive in a file-based stream.
    """
    start_offset: int
    compressed_size: int
    uncompressed_size: int
    compression_method: int

class ZipFileHandler:
    """
    Handler class for extracting information from ZIP files stored in AWS S3.
    """
    EOCD_SIGNATURE: bytes = b'PK\x05\x06'
    ZIP64_LOCATOR_SIGNATURE: bytes = b'PK\x06\x07'
    EOCD_CENTRAL_DIR_START_OFFSET: int = 16
    ZIP64_EOCD_OFFSET: int = 8
    ZIP64_EOCD_SIZE: int = 56
    ZIP64_CENTRAL_DIR_START_OFFSET: int = 48

    def __init__(self, s3_client: BaseClient, config: Config):
        if False:
            return 10
        '\n        Initialize the ZipFileHandler with an S3 client and configuration.\n\n        :param s3_client: The AWS S3 client.\n        :param config: Configuration containing bucket and other details.\n        '
        self.s3_client = s3_client
        self.config = config

    def _fetch_data_from_s3(self, filename: str, start: int, size: Optional[int]=None) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch a specific range of bytes from a file in S3.\n\n        :param filename: The name of the file in S3.\n        :param start: The starting byte position.\n        :param size: The number of bytes to fetch (optional).\n        :return: The fetched bytes.\n        '
        end_range = f'{start + size - 1}' if size else ''
        range_str = f'bytes={start}-{end_range}'
        response = self.s3_client.get_object(Bucket=self.config.bucket, Key=filename, Range=range_str)
        return response['Body'].read()

    def _find_signature(self, filename: str, signature: bytes, initial_buffer_size: int=BUFFER_SIZE_DEFAULT, max_buffer_size: int=MAX_BUFFER_SIZE_DEFAULT) -> Optional[bytes]:
        if False:
            print('Hello World!')
        '\n        Search for a specific signature in the file by checking chunks of increasing size.\n        If the signature is not found within the max_buffer_size, None is returned.\n\n        :param filename: The name of the file in S3.\n        :param signature: The byte signature to search for.\n        :param initial_buffer_size: Initial size of the buffer to search in.\n        :param max_buffer_size: Maximum size of the buffer to search in.\n        :return: The chunk of data containing the signature or None if not found.\n        '
        buffer_size = initial_buffer_size
        file_size = self.s3_client.head_object(Bucket=self.config.bucket, Key=filename)['ContentLength']
        while buffer_size <= max_buffer_size:
            chunk = self._fetch_data_from_s3(filename, file_size - buffer_size)
            index = chunk.rfind(signature)
            if index != -1:
                return chunk[index:]
            buffer_size *= 2
        return None

    def _fetch_zip64_data(self, filename: str) -> bytes:
        if False:
            print('Hello World!')
        '\n        Fetch the ZIP64 End of Central Directory (EOCD) data from a ZIP file.\n\n        :param filename: The name of the file in S3.\n        :return: The ZIP64 EOCD data.\n        '
        chunk = self._find_signature(filename, self.ZIP64_LOCATOR_SIGNATURE)
        zip64_eocd_offset = struct.unpack_from('<Q', chunk, self.ZIP64_EOCD_OFFSET)[0]
        return self._fetch_data_from_s3(filename, zip64_eocd_offset, self.ZIP64_EOCD_SIZE)

    def _get_central_directory_start(self, filename: str) -> int:
        if False:
            while True:
                i = 10
        '\n        Determine the starting position of the central directory in the ZIP file.\n        Adjusts for ZIP64 format if necessary.\n\n        :param filename: The name of the file in S3.\n        :return: The starting position of the central directory.\n        '
        eocd_data = self._find_signature(filename, self.EOCD_SIGNATURE)
        central_dir_start = struct.unpack_from('<L', eocd_data, self.EOCD_CENTRAL_DIR_START_OFFSET)[0]
        if central_dir_start == 4294967295:
            zip64_data = self._fetch_zip64_data(filename)
            central_dir_start = struct.unpack_from('<Q', zip64_data, self.ZIP64_CENTRAL_DIR_START_OFFSET)[0]
        return central_dir_start

    def get_zip_files(self, filename: str) -> Tuple[List[zipfile.ZipInfo], int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract metadata about the files inside a ZIP archive stored in S3.\n\n        :param filename: The name of the ZIP file in S3.\n        :return: A tuple containing a list of ZipInfo objects representing the files inside the ZIP archive\n                 and the starting position of the central directory.\n        '
        central_dir_start = self._get_central_directory_start(filename)
        central_dir_data = self._fetch_data_from_s3(filename, central_dir_start)
        with io.BytesIO(central_dir_data) as bytes_io:
            with zipfile.ZipFile(bytes_io, 'r') as zf:
                return (zf.infolist(), central_dir_start)

class DecompressedStream(io.IOBase):
    """
    A custom stream class that handles decompression of data from a given file object.
    This class supports seeking, reading, and other basic file operations on compressed data.
    """
    LOCAL_FILE_HEADER_SIZE: int = 30
    NAME_LENGTH_OFFSET: int = 26

    def __init__(self, file_obj: IO[bytes], file_info: RemoteFileInsideArchive, buffer_size: int=BUFFER_SIZE_DEFAULT):
        if False:
            i = 10
            return i + 15
        '\n        Initialize a DecompressedStream.\n\n        :param file_obj: Underlying file-like object.\n        :param file_info: Meta information about the file inside the archive.\n        :param buffer_size: Size of the buffer for reading data.\n        '
        self._file = file_obj
        self.file_start = self._calculate_actual_start(file_info.start_offset)
        self.compressed_size = file_info.compressed_size
        self.uncompressed_size = file_info.uncompressed_size
        self.compression_method = file_info.compression_method
        self._buffer = bytearray()
        self.buffer_size = buffer_size
        self._reset_decompressor()
        self.position = 0
        self._file.seek(self.file_start)
        self.offset_map = {0: self.file_start, self.uncompressed_size: self.file_start + self.compressed_size}

    def _calculate_actual_start(self, file_start: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        "\n        Determine the actual start position of the file content within the ZIP archive.\n\n        In a ZIP archive, each file entry is preceded by a local file header. This header contains\n        metadata about the file, including the lengths of the file's name and any extra data.\n        To accurately locate the start of the actual file content, we need to skip over this header.\n\n        This method calculates the start position by taking into account the length of the file name\n        and any extra data present in the local file header.\n\n        :param file_start: The starting position of the file entry (including its local file header)\n                           inside the ZIP archive.\n        :return: The actual starting position of the file content, after skipping the local file header.\n        "
        self._file.seek(file_start + self.NAME_LENGTH_OFFSET)
        (name_len, extra_len) = struct.unpack('<HH', self._file.read(4))
        return file_start + self.LOCAL_FILE_HEADER_SIZE + name_len + extra_len

    def _reset_decompressor(self):
        if False:
            i = 10
            return i + 15
        '\n        Reset the decompressor object.\n        '
        self.decompressor = zipfile._get_decompressor(self.compression_method)

    def _decompress_chunk(self, chunk: bytes) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        Decompress a chunk of data based on the compression method.\n        '
        if self.compression_method == zipfile.ZIP_STORED:
            return chunk
        return self.decompressor.decompress(chunk)

    def read(self, size: int=-1) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        '\n        Read a specified number of bytes from the stream.\n        '
        if size == -1:
            size = self.uncompressed_size - self.position
        if size <= len(self._buffer):
            data = self._buffer[:size]
            self._buffer = self._buffer[size:]
            self.position += len(data)
            return data
        data = self._buffer
        self._buffer = bytearray()
        while len(data) < size and self._file.tell() - self.file_start < self.compressed_size:
            max_read_size = min(self.buffer_size, self.compressed_size + self.file_start - self._file.tell())
            chunk = self._file.read(max_read_size)
            if not chunk:
                break
            decompressed_data = self._decompress_chunk(chunk)
            if len(data) + len(decompressed_data) > size:
                desired_length = size - len(data)
                data += decompressed_data[:desired_length]
                self._buffer = decompressed_data[desired_length:]
            else:
                data += decompressed_data
        self.position += len(data)
        return data

    def seek(self, offset: int, whence: int=io.SEEK_SET) -> int:
        if False:
            print('Hello World!')
        '\n        Seek to a specific position in the uncompressed stream.\n        '
        if whence == io.SEEK_SET:
            self._buffer = bytearray()
        elif whence == io.SEEK_CUR:
            offset = self.position + offset
        elif whence == io.SEEK_END:
            offset = self.uncompressed_size + offset
        offset = max(0, min(offset, self.uncompressed_size))
        closest_offset = max((k for k in self.offset_map if k <= offset))
        closest_position = self.offset_map[closest_offset]
        self._file.seek(closest_position)
        self._reset_decompressor()
        self.position = closest_offset
        while self.position < offset:
            read_size = min(self.buffer_size, offset - self.position)
            self.read(read_size)
        return self.position

    def tell(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Return the current position in the uncompressed stream.\n        '
        return self.position

    def readable(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return if the stream is readable.\n        '
        return True

    def seekable(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return if the stream is seekable.\n        '
        return True

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Close the stream and underlying file object.\n        '
        self._file.close()

class ZipContentReader:
    """
    A custom reader class that provides buffered reading capabilities on a decompressed stream.
    Supports reading lines, reading chunks, and iterating over the content.
    """

    def __init__(self, decompressed_stream: DecompressedStream, encoding: Optional[str]=None, buffer_size: int=BUFFER_SIZE_DEFAULT):
        if False:
            return 10
        '\n        Initialize a ZipContentReader.\n\n        :param decompressed_stream: A DecompressedStream object.\n        :param encoding: Encoding to decode the bytes. If None, bytes are returned.\n        :param buffer_size: Size of the buffer for reading data.\n        '
        self.raw = decompressed_stream
        self.encoding = encoding
        self.buffer_size = buffer_size
        self.buffer = bytearray()
        self._closed = False

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make the class iterable.\n        '
        return self

    def __next__(self) -> Union[str, bytes]:
        if False:
            while True:
                i = 10
        '\n        Iterate over the lines in the reader.\n        '
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def readline(self, limit: int=-1) -> Union[str, bytes]:
        if False:
            i = 10
            return i + 15
        '\n        Read a single line from the stream.\n        '
        if limit != -1:
            raise NotImplementedError('Limits other than -1 not implemented yet')
        line = ''
        while True:
            char = self.read(1)
            if not char:
                break
            line += char
            if char in ['\n', '\r']:
                next_char = self.read(1)
                if char == '\r' and next_char == '\n':
                    line += next_char
                else:
                    self.buffer = next_char.encode(self.encoding) + self.buffer
                break
        return line

    def read(self, size: int=-1) -> Union[str, bytes]:
        if False:
            print('Hello World!')
        '\n        Read a specified number of bytes/characters from the reader.\n        '
        while len(self.buffer) < size:
            chunk = self.raw.read(self.buffer_size)
            if not chunk:
                break
            self.buffer += chunk
        data = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return data.decode(self.encoding) if self.encoding else bytes(data)

    def seek(self, offset: int, whence: int=io.SEEK_SET) -> int:
        if False:
            print('Hello World!')
        '\n        Seek to a specific position in the decompressed stream.\n        '
        self.buffer = bytearray()
        return self.raw.seek(offset, whence)

    def close(self):
        if False:
            while True:
                i = 10
        '\n        Close the reader and underlying decompressed stream.\n        '
        self._closed = True
        self.raw.close()

    def tell(self) -> int:
        if False:
            return 10
        '\n        Return the current position in the decompressed stream.\n        '
        return self.raw.tell()

    @property
    def closed(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Check if the reader is closed.\n        '
        return self._closed

    def __enter__(self) -> 'ZipContentReader':
        if False:
            while True:
                i = 10
        'Enter the runtime context for the reader.'
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if False:
            return 10
        'Exit the runtime context for the reader and ensure resources are closed.'
        self.close()