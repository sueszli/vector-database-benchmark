"""Script comparing different pickling strategies."""
from joblib.numpy_pickle import NumpyPickler, NumpyUnpickler
from joblib.numpy_pickle_utils import BinaryZlibFile, BinaryGzipFile
from pickle import _Pickler, _Unpickler, Pickler, Unpickler
import numpy as np
import bz2
import lzma
import time
import io
import sys
import os
from collections import OrderedDict

def fileobj(obj, fname, mode, kwargs):
    if False:
        return 10
    'Create a file object.'
    return obj(fname, mode, **kwargs)

def bufferize(f, buf):
    if False:
        print('Hello World!')
    'Bufferize a fileobject using buf.'
    if buf is None:
        return f
    else:
        if buf.__name__ == io.BufferedWriter.__name__ or buf.__name__ == io.BufferedReader.__name__:
            return buf(f, buffer_size=10 * 1024 ** 2)
        return buf(f)

def _load(unpickler, fname, f):
    if False:
        i = 10
        return i + 15
    if unpickler.__name__ == NumpyUnpickler.__name__:
        p = unpickler(fname, f)
    else:
        p = unpickler(f)
    return p.load()

def print_line(obj, strategy, buffer, pickler, dump, load, disk_used):
    if False:
        for i in range(10):
            print('nop')
    'Nice printing function.'
    print('% 20s | %6s | % 14s | % 7s | % 5.1f\xa0| % 5.1f | % 5s' % (obj, strategy, buffer, pickler, dump, load, disk_used))

class PickleBufferedWriter:
    """Protect the underlying fileobj against numerous calls to write
    This is achieved by internally keeping a list of small chunks and
    only flushing to the backing fileobj if passed a large chunk or
    after a threshold on the number of small chunks.
    """

    def __init__(self, fileobj, max_buffer_size=10 * 1024 ** 2):
        if False:
            i = 10
            return i + 15
        self._fileobj = fileobj
        self._chunks = chunks = []

        def _write(data):
            if False:
                for i in range(10):
                    print('nop')
            chunks.append(data)
            if len(chunks) > max_buffer_size:
                self.flush()
        self.write = _write

    def flush(self):
        if False:
            while True:
                i = 10
        self._fileobj.write(b''.join(self._chunks[:]))
        del self._chunks[:]

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.flush()
        self._fileobj.close()

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, *exc):
        if False:
            return 10
        self.close()
        return False

class PickleBufferedReader:
    """Protect the underlying fileobj against numerous calls to write
    This is achieved by internally keeping a list of small chunks and
    only flushing to the backing fileobj if passed a large chunk or
    after a threshold on the number of small chunks.
    """

    def __init__(self, fileobj, max_buffer_size=10 * 1024 ** 2):
        if False:
            i = 10
            return i + 15
        self._fileobj = fileobj
        self._buffer = bytearray(max_buffer_size)
        self.max_buffer_size = max_buffer_size
        self._position = 0

    def read(self, n=None):
        if False:
            print('Hello World!')
        data = b''
        if n is None:
            data = self._fileobj.read()
        else:
            while len(data) < n:
                if self._position == 0:
                    self._buffer = self._fileobj.read(self.max_buffer_size)
                elif self._position == self.max_buffer_size:
                    self._position = 0
                    continue
                next_position = min(self.max_buffer_size, self._position + n - len(data))
                data += self._buffer[self._position:next_position]
                self._position = next_position
        return data

    def readline(self):
        if False:
            i = 10
            return i + 15
        line = []
        while True:
            c = self.read(1)
            line.append(c)
            if c == b'\n':
                break
        return b''.join(line)

    def close(self):
        if False:
            i = 10
            return i + 15
        self._fileobj.close()

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, *exc):
        if False:
            for i in range(10):
                print('nop')
        self.close()
        return False

def run_bench():
    if False:
        print('Hello World!')
    print('% 20s | %10s | % 12s | % 8s | % 9s | % 9s | % 5s' % ('Object', 'Compression', 'Buffer', 'Pickler/Unpickler', 'dump time (s)', 'load time (s)', 'Disk used (MB)'))
    print('--- | --- | --- | --- | --- | --- | ---')
    for (oname, obj) in objects.items():
        if isinstance(obj, np.ndarray):
            osize = obj.nbytes / 1000000.0
        else:
            osize = sys.getsizeof(obj) / 1000000.0
        for (cname, f) in compressors.items():
            fobj = f[0]
            fname = f[1]
            fmode = f[2]
            fopts = f[3]
            for (bname, buf) in bufs.items():
                writebuf = buf[0]
                readbuf = buf[1]
                for (pname, p) in picklers.items():
                    pickler = p[0]
                    unpickler = p[1]
                    t0 = time.time()
                    if writebuf is not None and writebuf.__name__ == io.BytesIO.__name__:
                        b = writebuf()
                        p = pickler(b)
                        p.dump(obj)
                        with fileobj(fobj, fname, fmode, fopts) as f:
                            f.write(b.getvalue())
                    else:
                        with bufferize(fileobj(fobj, fname, fmode, fopts), writebuf) as f:
                            p = pickler(f)
                            p.dump(obj)
                    dtime = time.time() - t0
                    t0 = time.time()
                    obj_r = None
                    if readbuf is not None and readbuf.__name__ == io.BytesIO.__name__:
                        b = readbuf()
                        with fileobj(fobj, fname, 'rb', {}) as f:
                            b.write(f.read())
                        b.seek(0)
                        obj_r = _load(unpickler, fname, b)
                    else:
                        with bufferize(fileobj(fobj, fname, 'rb', {}), readbuf) as f:
                            obj_r = _load(unpickler, fname, f)
                    ltime = time.time() - t0
                    if isinstance(obj, np.ndarray):
                        assert (obj == obj_r).all()
                    else:
                        assert obj == obj_r
                    print_line('{} ({:.1f}MB)'.format(oname, osize), cname, bname, pname, dtime, ltime, '{:.2f}'.format(os.path.getsize(fname) / 1000000.0))
DICT_SIZE = int(1000000.0)
ARRAY_SIZE = int(10000000.0)
arr = np.random.normal(size=ARRAY_SIZE)
arr[::2] = 1
objects = OrderedDict([('dict', dict(((i, str(i)) for i in range(DICT_SIZE)))), ('list', [i for i in range(DICT_SIZE)]), ('array semi-random', arr), ('array random', np.random.normal(size=ARRAY_SIZE)), ('array ones', np.ones(ARRAY_SIZE))])
picklers = OrderedDict([('Pickle', (_Pickler, _Unpickler)), ('cPickle', (Pickler, Unpickler)), ('Joblib', (NumpyPickler, NumpyUnpickler))])
compressors = OrderedDict([('No', (open, '/tmp/test_raw', 'wb', {})), ('Zlib', (BinaryZlibFile, '/tmp/test_zlib', 'wb', {'compresslevel': 3})), ('Gzip', (BinaryGzipFile, '/tmp/test_gzip', 'wb', {'compresslevel': 3})), ('Bz2', (bz2.BZ2File, '/tmp/test_bz2', 'wb', {'compresslevel': 3})), ('Xz', (lzma.LZMAFile, '/tmp/test_xz', 'wb', {'preset': 3, 'check': lzma.CHECK_NONE})), ('Lzma', (lzma.LZMAFile, '/tmp/test_lzma', 'wb', {'preset': 3, 'format': lzma.FORMAT_ALONE}))])
bufs = OrderedDict([('None', (None, None)), ('io.BytesIO', (io.BytesIO, io.BytesIO)), ('io.Buffered', (io.BufferedWriter, io.BufferedReader)), ('PickleBuffered', (PickleBufferedWriter, PickleBufferedReader))])
if __name__ == '__main__':
    run_bench()