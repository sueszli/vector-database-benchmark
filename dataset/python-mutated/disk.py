import gzip
from diskcache import FanoutCache, Disk
from diskcache.core import BytesType, MODE_BINARY, BytesIO
from util.logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class GzipDisk(Disk):

    def store(self, value, read, key=None):
        if False:
            return 10
        '\n        Override from base class diskcache.Disk.\n\n        Chunking is due to needing to work on pythons < 2.7.13:\n        - Issue #27130: In the "zlib" module, fix handling of large buffers\n          (typically 2 or 4 GiB).  Previously, inputs were limited to 2 GiB, and\n          compression and decompression operations did not properly handle results of\n          2 or 4 GiB.\n\n        :param value: value to convert\n        :param bool read: True when value is file-like object\n        :return: (size, mode, filename, value) tuple for Cache table\n        '
        if type(value) is BytesType:
            if read:
                value = value.read()
                read = False
            str_io = BytesIO()
            gz_file = gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io)
            for offset in range(0, len(value), 2 ** 30):
                gz_file.write(value[offset:offset + 2 ** 30])
            gz_file.close()
            value = str_io.getvalue()
        return super(GzipDisk, self).store(value, read)

    def fetch(self, mode, filename, value, read):
        if False:
            return 10
        '\n        Override from base class diskcache.Disk.\n\n        Chunking is due to needing to work on pythons < 2.7.13:\n        - Issue #27130: In the "zlib" module, fix handling of large buffers\n          (typically 2 or 4 GiB).  Previously, inputs were limited to 2 GiB, and\n          compression and decompression operations did not properly handle results of\n          2 or 4 GiB.\n\n        :param int mode: value mode raw, binary, text, or pickle\n        :param str filename: filename of corresponding value\n        :param value: database value\n        :param bool read: when True, return an open file handle\n        :return: corresponding Python value\n        '
        value = super(GzipDisk, self).fetch(mode, filename, value, read)
        if mode == MODE_BINARY:
            str_io = BytesIO(value)
            gz_file = gzip.GzipFile(mode='rb', fileobj=str_io)
            read_csio = BytesIO()
            while True:
                uncompressed_data = gz_file.read(2 ** 30)
                if uncompressed_data:
                    read_csio.write(uncompressed_data)
                else:
                    break
            value = read_csio.getvalue()
        return value

def getCache(scope_str):
    if False:
        print('Hello World!')
    return FanoutCache('data-unversioned/cache/' + scope_str, disk=GzipDisk, shards=64, timeout=1, size_limit=300000000000.0)