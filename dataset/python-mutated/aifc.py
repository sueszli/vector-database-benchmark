"""Stuff to parse AIFF-C and AIFF files.

Unless explicitly stated otherwise, the description below is true
both for AIFF-C files and AIFF files.

An AIFF-C file has the following structure.

  +-----------------+
  | FORM            |
  +-----------------+
  | <size>          |
  +----+------------+
  |    | AIFC       |
  |    +------------+
  |    | <chunks>   |
  |    |    .       |
  |    |    .       |
  |    |    .       |
  +----+------------+

An AIFF file has the string "AIFF" instead of "AIFC".

A chunk consists of an identifier (4 bytes) followed by a size (4 bytes,
big endian order), followed by the data.  The size field does not include
the size of the 8 byte header.

The following chunk types are recognized.

  FVER
      <version number of AIFF-C defining document> (AIFF-C only).
  MARK
      <# of markers> (2 bytes)
      list of markers:
          <marker ID> (2 bytes, must be > 0)
          <position> (4 bytes)
          <marker name> ("pstring")
  COMM
      <# of channels> (2 bytes)
      <# of sound frames> (4 bytes)
      <size of the samples> (2 bytes)
      <sampling frequency> (10 bytes, IEEE 80-bit extended
          floating point)
      in AIFF-C files only:
      <compression type> (4 bytes)
      <human-readable version of compression type> ("pstring")
  SSND
      <offset> (4 bytes, not used by this program)
      <blocksize> (4 bytes, not used by this program)
      <sound data>

A pstring consists of 1 byte length, a string of characters, and 0 or 1
byte pad to make the total length even.

Usage.

Reading AIFF files:
  f = aifc.open(file, 'r')
where file is either the name of a file or an open file pointer.
The open file pointer must have methods read(), seek(), and close().
In some types of audio files, if the setpos() method is not used,
the seek() method is not necessary.

This returns an instance of a class with the following public methods:
  getnchannels()  -- returns number of audio channels (1 for
             mono, 2 for stereo)
  getsampwidth()  -- returns sample width in bytes
  getframerate()  -- returns sampling frequency
  getnframes()    -- returns number of audio frames
  getcomptype()   -- returns compression type ('NONE' for AIFF files)
  getcompname()   -- returns human-readable version of
             compression type ('not compressed' for AIFF files)
  getparams() -- returns a namedtuple consisting of all of the
             above in the above order
  getmarkers()    -- get the list of marks in the audio file or None
             if there are no marks
  getmark(id) -- get mark with the specified id (raises an error
             if the mark does not exist)
  readframes(n)   -- returns at most n frames of audio
  rewind()    -- rewind to the beginning of the audio stream
  setpos(pos) -- seek to the specified position
  tell()      -- return the current position
  close()     -- close the instance (make it unusable)
The position returned by tell(), the position given to setpos() and
the position of marks are all compatible and have nothing to do with
the actual position in the file.
The close() method is called automatically when the class instance
is destroyed.

Writing AIFF files:
  f = aifc.open(file, 'w')
where file is either the name of a file or an open file pointer.
The open file pointer must have methods write(), tell(), seek(), and
close().

This returns an instance of a class with the following public methods:
  aiff()      -- create an AIFF file (AIFF-C default)
  aifc()      -- create an AIFF-C file
  setnchannels(n) -- set the number of channels
  setsampwidth(n) -- set the sample width
  setframerate(n) -- set the frame rate
  setnframes(n)   -- set the number of frames
  setcomptype(type, name)
          -- set the compression type and the
             human-readable compression type
  setparams(tuple)
          -- set all parameters at once
  setmark(id, pos, name)
          -- add specified mark to the list of marks
  tell()      -- return current position in output file (useful
             in combination with setmark())
  writeframesraw(data)
          -- write audio frames without pathing up the
             file header
  writeframes(data)
          -- write audio frames and patch up the file header
  close()     -- patch up the file header and close the
             output file
You should set the parameters before the first writeframesraw or
writeframes.  The total number of frames does not need to be set,
but when it is set to the correct value, the header does not have to
be patched up.
It is best to first set all parameters, perhaps possibly the
compression type, and then write audio frames using writeframesraw.
When all frames have been written, either call writeframes(b'') or
close() to patch up the sizes in the header.
Marks can be added anytime.  If there are any marks, you must call
close() after all frames have been written.
The close() method is called automatically when the class instance
is destroyed.

When a file is opened with the extension '.aiff', an AIFF file is
written, otherwise an AIFF-C file is written.  This default can be
changed by calling aiff() or aifc() before the first writeframes or
writeframesraw.
"""
import struct
import builtins
import warnings
__all__ = ['Error', 'open']

class Error(Exception):
    pass
_AIFC_version = 2726318400

def _read_long(file):
    if False:
        i = 10
        return i + 15
    try:
        return struct.unpack('>l', file.read(4))[0]
    except struct.error:
        raise EOFError from None

def _read_ulong(file):
    if False:
        for i in range(10):
            print('nop')
    try:
        return struct.unpack('>L', file.read(4))[0]
    except struct.error:
        raise EOFError from None

def _read_short(file):
    if False:
        while True:
            i = 10
    try:
        return struct.unpack('>h', file.read(2))[0]
    except struct.error:
        raise EOFError from None

def _read_ushort(file):
    if False:
        for i in range(10):
            print('nop')
    try:
        return struct.unpack('>H', file.read(2))[0]
    except struct.error:
        raise EOFError from None

def _read_string(file):
    if False:
        return 10
    length = ord(file.read(1))
    if length == 0:
        data = b''
    else:
        data = file.read(length)
    if length & 1 == 0:
        dummy = file.read(1)
    return data
_HUGE_VAL = 1.79769313486231e+308

def _read_float(f):
    if False:
        i = 10
        return i + 15
    expon = _read_short(f)
    sign = 1
    if expon < 0:
        sign = -1
        expon = expon + 32768
    himant = _read_ulong(f)
    lomant = _read_ulong(f)
    if expon == himant == lomant == 0:
        f = 0.0
    elif expon == 32767:
        f = _HUGE_VAL
    else:
        expon = expon - 16383
        f = (himant * 4294967296 + lomant) * pow(2.0, expon - 63)
    return sign * f

def _write_short(f, x):
    if False:
        for i in range(10):
            print('nop')
    f.write(struct.pack('>h', x))

def _write_ushort(f, x):
    if False:
        return 10
    f.write(struct.pack('>H', x))

def _write_long(f, x):
    if False:
        for i in range(10):
            print('nop')
    f.write(struct.pack('>l', x))

def _write_ulong(f, x):
    if False:
        while True:
            i = 10
    f.write(struct.pack('>L', x))

def _write_string(f, s):
    if False:
        for i in range(10):
            print('nop')
    if len(s) > 255:
        raise ValueError('string exceeds maximum pstring length')
    f.write(struct.pack('B', len(s)))
    f.write(s)
    if len(s) & 1 == 0:
        f.write(b'\x00')

def _write_float(f, x):
    if False:
        print('Hello World!')
    import math
    if x < 0:
        sign = 32768
        x = x * -1
    else:
        sign = 0
    if x == 0:
        expon = 0
        himant = 0
        lomant = 0
    else:
        (fmant, expon) = math.frexp(x)
        if expon > 16384 or fmant >= 1 or fmant != fmant:
            expon = sign | 32767
            himant = 0
            lomant = 0
        else:
            expon = expon + 16382
            if expon < 0:
                fmant = math.ldexp(fmant, expon)
                expon = 0
            expon = expon | sign
            fmant = math.ldexp(fmant, 32)
            fsmant = math.floor(fmant)
            himant = int(fsmant)
            fmant = math.ldexp(fmant - fsmant, 32)
            fsmant = math.floor(fmant)
            lomant = int(fsmant)
    _write_ushort(f, expon)
    _write_ulong(f, himant)
    _write_ulong(f, lomant)
from chunk import Chunk
from collections import namedtuple
_aifc_params = namedtuple('_aifc_params', 'nchannels sampwidth framerate nframes comptype compname')
_aifc_params.nchannels.__doc__ = 'Number of audio channels (1 for mono, 2 for stereo)'
_aifc_params.sampwidth.__doc__ = 'Sample width in bytes'
_aifc_params.framerate.__doc__ = 'Sampling frequency'
_aifc_params.nframes.__doc__ = 'Number of audio frames'
_aifc_params.comptype.__doc__ = 'Compression type ("NONE" for AIFF files)'
_aifc_params.compname.__doc__ = "A human-readable version of the compression type\n('not compressed' for AIFF files)"

class Aifc_read:
    _file = None

    def initfp(self, file):
        if False:
            for i in range(10):
                print('nop')
        self._version = 0
        self._convert = None
        self._markers = []
        self._soundpos = 0
        self._file = file
        chunk = Chunk(file)
        if chunk.getname() != b'FORM':
            raise Error('file does not start with FORM id')
        formdata = chunk.read(4)
        if formdata == b'AIFF':
            self._aifc = 0
        elif formdata == b'AIFC':
            self._aifc = 1
        else:
            raise Error('not an AIFF or AIFF-C file')
        self._comm_chunk_read = 0
        self._ssnd_chunk = None
        while 1:
            self._ssnd_seek_needed = 1
            try:
                chunk = Chunk(self._file)
            except EOFError:
                break
            chunkname = chunk.getname()
            if chunkname == b'COMM':
                self._read_comm_chunk(chunk)
                self._comm_chunk_read = 1
            elif chunkname == b'SSND':
                self._ssnd_chunk = chunk
                dummy = chunk.read(8)
                self._ssnd_seek_needed = 0
            elif chunkname == b'FVER':
                self._version = _read_ulong(chunk)
            elif chunkname == b'MARK':
                self._readmark(chunk)
            chunk.skip()
        if not self._comm_chunk_read or not self._ssnd_chunk:
            raise Error('COMM chunk and/or SSND chunk missing')

    def __init__(self, f):
        if False:
            return 10
        if isinstance(f, str):
            file_object = builtins.open(f, 'rb')
            try:
                self.initfp(file_object)
            except:
                file_object.close()
                raise
        else:
            self.initfp(f)

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        self.close()

    def getfp(self):
        if False:
            print('Hello World!')
        return self._file

    def rewind(self):
        if False:
            print('Hello World!')
        self._ssnd_seek_needed = 1
        self._soundpos = 0

    def close(self):
        if False:
            while True:
                i = 10
        file = self._file
        if file is not None:
            self._file = None
            file.close()

    def tell(self):
        if False:
            for i in range(10):
                print('nop')
        return self._soundpos

    def getnchannels(self):
        if False:
            for i in range(10):
                print('nop')
        return self._nchannels

    def getnframes(self):
        if False:
            print('Hello World!')
        return self._nframes

    def getsampwidth(self):
        if False:
            while True:
                i = 10
        return self._sampwidth

    def getframerate(self):
        if False:
            print('Hello World!')
        return self._framerate

    def getcomptype(self):
        if False:
            print('Hello World!')
        return self._comptype

    def getcompname(self):
        if False:
            print('Hello World!')
        return self._compname

    def getparams(self):
        if False:
            while True:
                i = 10
        return _aifc_params(self.getnchannels(), self.getsampwidth(), self.getframerate(), self.getnframes(), self.getcomptype(), self.getcompname())

    def getmarkers(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self._markers) == 0:
            return None
        return self._markers

    def getmark(self, id):
        if False:
            i = 10
            return i + 15
        for marker in self._markers:
            if id == marker[0]:
                return marker
        raise Error('marker {0!r} does not exist'.format(id))

    def setpos(self, pos):
        if False:
            while True:
                i = 10
        if pos < 0 or pos > self._nframes:
            raise Error('position not in range')
        self._soundpos = pos
        self._ssnd_seek_needed = 1

    def readframes(self, nframes):
        if False:
            while True:
                i = 10
        if self._ssnd_seek_needed:
            self._ssnd_chunk.seek(0)
            dummy = self._ssnd_chunk.read(8)
            pos = self._soundpos * self._framesize
            if pos:
                self._ssnd_chunk.seek(pos + 8)
            self._ssnd_seek_needed = 0
        if nframes == 0:
            return b''
        data = self._ssnd_chunk.read(nframes * self._framesize)
        if self._convert and data:
            data = self._convert(data)
        self._soundpos = self._soundpos + len(data) // (self._nchannels * self._sampwidth)
        return data

    def _alaw2lin(self, data):
        if False:
            print('Hello World!')
        import audioop
        return audioop.alaw2lin(data, 2)

    def _ulaw2lin(self, data):
        if False:
            print('Hello World!')
        import audioop
        return audioop.ulaw2lin(data, 2)

    def _adpcm2lin(self, data):
        if False:
            i = 10
            return i + 15
        import audioop
        if not hasattr(self, '_adpcmstate'):
            self._adpcmstate = None
        (data, self._adpcmstate) = audioop.adpcm2lin(data, 2, self._adpcmstate)
        return data

    def _read_comm_chunk(self, chunk):
        if False:
            while True:
                i = 10
        self._nchannels = _read_short(chunk)
        self._nframes = _read_long(chunk)
        self._sampwidth = (_read_short(chunk) + 7) // 8
        self._framerate = int(_read_float(chunk))
        if self._sampwidth <= 0:
            raise Error('bad sample width')
        if self._nchannels <= 0:
            raise Error('bad # of channels')
        self._framesize = self._nchannels * self._sampwidth
        if self._aifc:
            kludge = 0
            if chunk.chunksize == 18:
                kludge = 1
                warnings.warn('Warning: bad COMM chunk size')
                chunk.chunksize = 23
            self._comptype = chunk.read(4)
            if kludge:
                length = ord(chunk.file.read(1))
                if length & 1 == 0:
                    length = length + 1
                chunk.chunksize = chunk.chunksize + length
                chunk.file.seek(-1, 1)
            self._compname = _read_string(chunk)
            if self._comptype != b'NONE':
                if self._comptype == b'G722':
                    self._convert = self._adpcm2lin
                elif self._comptype in (b'ulaw', b'ULAW'):
                    self._convert = self._ulaw2lin
                elif self._comptype in (b'alaw', b'ALAW'):
                    self._convert = self._alaw2lin
                else:
                    raise Error('unsupported compression type')
                self._sampwidth = 2
        else:
            self._comptype = b'NONE'
            self._compname = b'not compressed'

    def _readmark(self, chunk):
        if False:
            for i in range(10):
                print('nop')
        nmarkers = _read_short(chunk)
        try:
            for i in range(nmarkers):
                id = _read_short(chunk)
                pos = _read_long(chunk)
                name = _read_string(chunk)
                if pos or name:
                    self._markers.append((id, pos, name))
        except EOFError:
            w = 'Warning: MARK chunk contains only %s marker%s instead of %s' % (len(self._markers), '' if len(self._markers) == 1 else 's', nmarkers)
            warnings.warn(w)

class Aifc_write:
    _file = None

    def __init__(self, f):
        if False:
            i = 10
            return i + 15
        if isinstance(f, str):
            file_object = builtins.open(f, 'wb')
            try:
                self.initfp(file_object)
            except:
                file_object.close()
                raise
            if f.endswith('.aiff'):
                self._aifc = 0
        else:
            self.initfp(f)

    def initfp(self, file):
        if False:
            return 10
        self._file = file
        self._version = _AIFC_version
        self._comptype = b'NONE'
        self._compname = b'not compressed'
        self._convert = None
        self._nchannels = 0
        self._sampwidth = 0
        self._framerate = 0
        self._nframes = 0
        self._nframeswritten = 0
        self._datawritten = 0
        self._datalength = 0
        self._markers = []
        self._marklength = 0
        self._aifc = 1

    def __del__(self):
        if False:
            print('Hello World!')
        self.close()

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        self.close()

    def aiff(self):
        if False:
            return 10
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._aifc = 0

    def aifc(self):
        if False:
            while True:
                i = 10
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._aifc = 1

    def setnchannels(self, nchannels):
        if False:
            for i in range(10):
                print('nop')
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if nchannels < 1:
            raise Error('bad # of channels')
        self._nchannels = nchannels

    def getnchannels(self):
        if False:
            while True:
                i = 10
        if not self._nchannels:
            raise Error('number of channels not set')
        return self._nchannels

    def setsampwidth(self, sampwidth):
        if False:
            i = 10
            return i + 15
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if sampwidth < 1 or sampwidth > 4:
            raise Error('bad sample width')
        self._sampwidth = sampwidth

    def getsampwidth(self):
        if False:
            while True:
                i = 10
        if not self._sampwidth:
            raise Error('sample width not set')
        return self._sampwidth

    def setframerate(self, framerate):
        if False:
            while True:
                i = 10
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if framerate <= 0:
            raise Error('bad frame rate')
        self._framerate = framerate

    def getframerate(self):
        if False:
            print('Hello World!')
        if not self._framerate:
            raise Error('frame rate not set')
        return self._framerate

    def setnframes(self, nframes):
        if False:
            i = 10
            return i + 15
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._nframes = nframes

    def getnframes(self):
        if False:
            for i in range(10):
                print('nop')
        return self._nframeswritten

    def setcomptype(self, comptype, compname):
        if False:
            for i in range(10):
                print('nop')
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if comptype not in (b'NONE', b'ulaw', b'ULAW', b'alaw', b'ALAW', b'G722'):
            raise Error('unsupported compression type')
        self._comptype = comptype
        self._compname = compname

    def getcomptype(self):
        if False:
            while True:
                i = 10
        return self._comptype

    def getcompname(self):
        if False:
            for i in range(10):
                print('nop')
        return self._compname

    def setparams(self, params):
        if False:
            i = 10
            return i + 15
        (nchannels, sampwidth, framerate, nframes, comptype, compname) = params
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if comptype not in (b'NONE', b'ulaw', b'ULAW', b'alaw', b'ALAW', b'G722'):
            raise Error('unsupported compression type')
        self.setnchannels(nchannels)
        self.setsampwidth(sampwidth)
        self.setframerate(framerate)
        self.setnframes(nframes)
        self.setcomptype(comptype, compname)

    def getparams(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._nchannels or not self._sampwidth or (not self._framerate):
            raise Error('not all parameters set')
        return _aifc_params(self._nchannels, self._sampwidth, self._framerate, self._nframes, self._comptype, self._compname)

    def setmark(self, id, pos, name):
        if False:
            i = 10
            return i + 15
        if id <= 0:
            raise Error('marker ID must be > 0')
        if pos < 0:
            raise Error('marker position must be >= 0')
        if not isinstance(name, bytes):
            raise Error('marker name must be bytes')
        for i in range(len(self._markers)):
            if id == self._markers[i][0]:
                self._markers[i] = (id, pos, name)
                return
        self._markers.append((id, pos, name))

    def getmark(self, id):
        if False:
            for i in range(10):
                print('nop')
        for marker in self._markers:
            if id == marker[0]:
                return marker
        raise Error('marker {0!r} does not exist'.format(id))

    def getmarkers(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self._markers) == 0:
            return None
        return self._markers

    def tell(self):
        if False:
            i = 10
            return i + 15
        return self._nframeswritten

    def writeframesraw(self, data):
        if False:
            return 10
        if not isinstance(data, (bytes, bytearray)):
            data = memoryview(data).cast('B')
        self._ensure_header_written(len(data))
        nframes = len(data) // (self._sampwidth * self._nchannels)
        if self._convert:
            data = self._convert(data)
        self._file.write(data)
        self._nframeswritten = self._nframeswritten + nframes
        self._datawritten = self._datawritten + len(data)

    def writeframes(self, data):
        if False:
            while True:
                i = 10
        self.writeframesraw(data)
        if self._nframeswritten != self._nframes or self._datalength != self._datawritten:
            self._patchheader()

    def close(self):
        if False:
            return 10
        if self._file is None:
            return
        try:
            self._ensure_header_written(0)
            if self._datawritten & 1:
                self._file.write(b'\x00')
                self._datawritten = self._datawritten + 1
            self._writemarkers()
            if self._nframeswritten != self._nframes or self._datalength != self._datawritten or self._marklength:
                self._patchheader()
        finally:
            self._convert = None
            f = self._file
            self._file = None
            f.close()

    def _lin2alaw(self, data):
        if False:
            while True:
                i = 10
        import audioop
        return audioop.lin2alaw(data, 2)

    def _lin2ulaw(self, data):
        if False:
            i = 10
            return i + 15
        import audioop
        return audioop.lin2ulaw(data, 2)

    def _lin2adpcm(self, data):
        if False:
            while True:
                i = 10
        import audioop
        if not hasattr(self, '_adpcmstate'):
            self._adpcmstate = None
        (data, self._adpcmstate) = audioop.lin2adpcm(data, 2, self._adpcmstate)
        return data

    def _ensure_header_written(self, datasize):
        if False:
            print('Hello World!')
        if not self._nframeswritten:
            if self._comptype in (b'ULAW', b'ulaw', b'ALAW', b'alaw', b'G722'):
                if not self._sampwidth:
                    self._sampwidth = 2
                if self._sampwidth != 2:
                    raise Error('sample width must be 2 when compressing with ulaw/ULAW, alaw/ALAW or G7.22 (ADPCM)')
            if not self._nchannels:
                raise Error('# channels not specified')
            if not self._sampwidth:
                raise Error('sample width not specified')
            if not self._framerate:
                raise Error('sampling rate not specified')
            self._write_header(datasize)

    def _init_compression(self):
        if False:
            return 10
        if self._comptype == b'G722':
            self._convert = self._lin2adpcm
        elif self._comptype in (b'ulaw', b'ULAW'):
            self._convert = self._lin2ulaw
        elif self._comptype in (b'alaw', b'ALAW'):
            self._convert = self._lin2alaw

    def _write_header(self, initlength):
        if False:
            i = 10
            return i + 15
        if self._aifc and self._comptype != b'NONE':
            self._init_compression()
        self._file.write(b'FORM')
        if not self._nframes:
            self._nframes = initlength // (self._nchannels * self._sampwidth)
        self._datalength = self._nframes * self._nchannels * self._sampwidth
        if self._datalength & 1:
            self._datalength = self._datalength + 1
        if self._aifc:
            if self._comptype in (b'ulaw', b'ULAW', b'alaw', b'ALAW'):
                self._datalength = self._datalength // 2
                if self._datalength & 1:
                    self._datalength = self._datalength + 1
            elif self._comptype == b'G722':
                self._datalength = (self._datalength + 3) // 4
                if self._datalength & 1:
                    self._datalength = self._datalength + 1
        try:
            self._form_length_pos = self._file.tell()
        except (AttributeError, OSError):
            self._form_length_pos = None
        commlength = self._write_form_length(self._datalength)
        if self._aifc:
            self._file.write(b'AIFC')
            self._file.write(b'FVER')
            _write_ulong(self._file, 4)
            _write_ulong(self._file, self._version)
        else:
            self._file.write(b'AIFF')
        self._file.write(b'COMM')
        _write_ulong(self._file, commlength)
        _write_short(self._file, self._nchannels)
        if self._form_length_pos is not None:
            self._nframes_pos = self._file.tell()
        _write_ulong(self._file, self._nframes)
        if self._comptype in (b'ULAW', b'ulaw', b'ALAW', b'alaw', b'G722'):
            _write_short(self._file, 8)
        else:
            _write_short(self._file, self._sampwidth * 8)
        _write_float(self._file, self._framerate)
        if self._aifc:
            self._file.write(self._comptype)
            _write_string(self._file, self._compname)
        self._file.write(b'SSND')
        if self._form_length_pos is not None:
            self._ssnd_length_pos = self._file.tell()
        _write_ulong(self._file, self._datalength + 8)
        _write_ulong(self._file, 0)
        _write_ulong(self._file, 0)

    def _write_form_length(self, datalength):
        if False:
            return 10
        if self._aifc:
            commlength = 18 + 5 + len(self._compname)
            if commlength & 1:
                commlength = commlength + 1
            verslength = 12
        else:
            commlength = 18
            verslength = 0
        _write_ulong(self._file, 4 + verslength + self._marklength + 8 + commlength + 16 + datalength)
        return commlength

    def _patchheader(self):
        if False:
            while True:
                i = 10
        curpos = self._file.tell()
        if self._datawritten & 1:
            datalength = self._datawritten + 1
            self._file.write(b'\x00')
        else:
            datalength = self._datawritten
        if datalength == self._datalength and self._nframes == self._nframeswritten and (self._marklength == 0):
            self._file.seek(curpos, 0)
            return
        self._file.seek(self._form_length_pos, 0)
        dummy = self._write_form_length(datalength)
        self._file.seek(self._nframes_pos, 0)
        _write_ulong(self._file, self._nframeswritten)
        self._file.seek(self._ssnd_length_pos, 0)
        _write_ulong(self._file, datalength + 8)
        self._file.seek(curpos, 0)
        self._nframes = self._nframeswritten
        self._datalength = datalength

    def _writemarkers(self):
        if False:
            while True:
                i = 10
        if len(self._markers) == 0:
            return
        self._file.write(b'MARK')
        length = 2
        for marker in self._markers:
            (id, pos, name) = marker
            length = length + len(name) + 1 + 6
            if len(name) & 1 == 0:
                length = length + 1
        _write_ulong(self._file, length)
        self._marklength = length + 8
        _write_short(self._file, len(self._markers))
        for marker in self._markers:
            (id, pos, name) = marker
            _write_short(self._file, id)
            _write_ulong(self._file, pos)
            _write_string(self._file, name)

def open(f, mode=None):
    if False:
        while True:
            i = 10
    if mode is None:
        if hasattr(f, 'mode'):
            mode = f.mode
        else:
            mode = 'rb'
    if mode in ('r', 'rb'):
        return Aifc_read(f)
    elif mode in ('w', 'wb'):
        return Aifc_write(f)
    else:
        raise Error("mode must be 'r', 'rb', 'w', or 'wb'")
if __name__ == '__main__':
    import sys
    if not sys.argv[1:]:
        sys.argv.append('/usr/demos/data/audio/bach.aiff')
    fn = sys.argv[1]
    with open(fn, 'r') as f:
        print('Reading', fn)
        print('nchannels =', f.getnchannels())
        print('nframes   =', f.getnframes())
        print('sampwidth =', f.getsampwidth())
        print('framerate =', f.getframerate())
        print('comptype  =', f.getcomptype())
        print('compname  =', f.getcompname())
        if sys.argv[2:]:
            gn = sys.argv[2]
            print('Writing', gn)
            with open(gn, 'w') as g:
                g.setparams(f.getparams())
                while 1:
                    data = f.readframes(1024)
                    if not data:
                        break
                    g.writeframes(data)
            print('Done.')