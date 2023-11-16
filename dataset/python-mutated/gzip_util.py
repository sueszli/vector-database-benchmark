"""
    salt.utils.gzip
    ~~~~~~~~~~~~~~~
    Helper module for handling gzip consistently between 2.7+ and 2.6-
"""
import gzip
import io
import salt.utils.files

class GzipFile(gzip.GzipFile):

    def __init__(self, filename=None, mode=None, compresslevel=9, fileobj=None):
        if False:
            return 10
        gzip.GzipFile.__init__(self, filename, mode, compresslevel, fileobj)

    def __enter__(self):
        if False:
            print('Hello World!')
        'Context management protocol.  Returns self.'
        return self

    def __exit__(self, *args):
        if False:
            return 10
        'Context management protocol.  Calls close()'
        self.close()

def open(filename, mode='rb', compresslevel=9):
    if False:
        print('Hello World!')
    if hasattr(gzip.GzipFile, '__enter__'):
        return gzip.open(filename, mode, compresslevel)
    else:
        return GzipFile(filename, mode, compresslevel)

def open_fileobj(fileobj, mode='rb', compresslevel=9):
    if False:
        print('Hello World!')
    if hasattr(gzip.GzipFile, '__enter__'):
        return gzip.GzipFile(filename='', mode=mode, fileobj=fileobj, compresslevel=compresslevel)
    return GzipFile(filename='', mode=mode, fileobj=fileobj, compresslevel=compresslevel)

def compress(data, compresslevel=9):
    if False:
        while True:
            i = 10
    '\n    Returns the data compressed at gzip level compression.\n    '
    buf = io.BytesIO()
    with open_fileobj(buf, 'wb', compresslevel) as ogz:
        if not isinstance(data, bytes):
            data = data.encode(__salt_system_encoding__)
        ogz.write(data)
    compressed = buf.getvalue()
    return compressed

def uncompress(data):
    if False:
        for i in range(10):
            print('nop')
    buf = io.BytesIO(data)
    with open_fileobj(buf, 'rb') as igz:
        unc = igz.read()
        return unc

def compress_file(fh_, compresslevel=9, chunk_size=1048576):
    if False:
        i = 10
        return i + 15
    '\n    Generator that reads chunk_size bytes at a time from a file/filehandle and\n    yields the compressed result of each read.\n\n    .. note::\n        Each chunk is compressed separately. They cannot be stitched together\n        to form a compressed file. This function is designed to break up a file\n        into compressed chunks for transport and decompression/reassembly on a\n        remote host.\n    '
    try:
        bytes_read = int(chunk_size)
        if bytes_read != chunk_size:
            raise ValueError
    except ValueError:
        raise ValueError('chunk_size must be an integer')
    try:
        while bytes_read == chunk_size:
            buf = io.BytesIO()
            with open_fileobj(buf, 'wb', compresslevel) as ogz:
                try:
                    bytes_read = ogz.write(fh_.read(chunk_size))
                except AttributeError:
                    fh_ = salt.utils.files.fopen(fh_, 'rb')
                    bytes_read = ogz.write(fh_.read(chunk_size))
            yield buf.getvalue()
    finally:
        try:
            fh_.close()
        except AttributeError:
            pass