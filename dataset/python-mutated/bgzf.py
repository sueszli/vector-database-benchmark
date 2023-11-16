"""Read and write BGZF compressed files (the GZIP variant used in BAM).

The SAM/BAM file format (Sequence Alignment/Map) comes in a plain text
format (SAM), and a compressed binary format (BAM). The latter uses a
modified form of gzip compression called BGZF (Blocked GNU Zip Format),
which can be applied to any file format to provide compression with
efficient random access. BGZF is described together with the SAM/BAM
file format at https://samtools.sourceforge.net/SAM1.pdf

Please read the text below about 'virtual offsets' before using BGZF
files for random access.

Aim of this module
------------------
The Python gzip library can be used to read BGZF files, since for
decompression they are just (specialised) gzip files. What this
module aims to facilitate is random access to BGZF files (using the
'virtual offset' idea), and writing BGZF files (which means using
suitably sized gzip blocks and writing the extra 'BC' field in the
gzip headers). As in the gzip library, the zlib library is used
internally.

In addition to being required for random access to and writing of
BAM files, the BGZF format can also be used on other sequential
data (in the sense of one record after another), such as most of
the sequence data formats supported in Bio.SeqIO (like FASTA,
FASTQ, GenBank, etc) or large MAF alignments.

The Bio.SeqIO indexing functions use this module to support BGZF files.

Technical Introduction to BGZF
------------------------------
The gzip file format allows multiple compressed blocks, each of which
could be a stand alone gzip file. As an interesting bonus, this means
you can use Unix ``cat`` to combine two or more gzip files into one by
concatenating them. Also, each block can have one of several compression
levels (including uncompressed, which actually takes up a little bit
more space due to the gzip header).

What the BAM designers realised was that while random access to data
stored in traditional gzip files was slow, breaking the file into
gzip blocks would allow fast random access to each block. To access
a particular piece of the decompressed data, you just need to know
which block it starts in (the offset of the gzip block start), and
how far into the (decompressed) contents of the block you need to
read.

One problem with this is finding the gzip block sizes efficiently.
You can do it with a standard gzip file, but it requires every block
to be decompressed -- and that would be rather slow. Additionally
typical gzip files may use very large blocks.

All that differs in BGZF is that compressed size of each gzip block
is limited to 2^16 bytes, and an extra 'BC' field in the gzip header
records this size. Traditional decompression tools can ignore this,
and unzip the file just like any other gzip file.

The point of this is you can look at the first BGZF block, find out
how big it is from this 'BC' header, and thus seek immediately to
the second block, and so on.

The BAM indexing scheme records read positions using a 64 bit
'virtual offset', comprising ``coffset << 16 | uoffset``, where ``coffset``
is the file offset of the BGZF block containing the start of the read
(unsigned integer using up to 64-16 = 48 bits), and ``uoffset`` is the
offset within the (decompressed) block (unsigned 16 bit integer).

This limits you to BAM files where the last block starts by 2^48
bytes, or 256 petabytes, and the decompressed size of each block
is at most 2^16 bytes, or 64kb. Note that this matches the BGZF
'BC' field size which limits the compressed size of each block to
2^16 bytes, allowing for BAM files to use BGZF with no gzip
compression (useful for intermediate files in memory to reduce
CPU load).

Warning about namespaces
------------------------
It is considered a bad idea to use "from XXX import ``*``" in Python, because
it pollutes the namespace. This is a real issue with Bio.bgzf (and the
standard Python library gzip) because they contain a function called open
i.e. Suppose you do this:

>>> from Bio.bgzf import *
>>> print(open.__module__)
Bio.bgzf

Or,

>>> from gzip import *
>>> print(open.__module__)
gzip

Notice that the open function has been replaced. You can "fix" this if you
need to by importing the built-in open function:

>>> from builtins import open

However, what we recommend instead is to use the explicit namespace, e.g.

>>> from Bio import bgzf
>>> print(bgzf.open.__module__)
Bio.bgzf


Examples
--------
This is an ordinary GenBank file compressed using BGZF, so it can
be decompressed using gzip,

>>> import gzip
>>> handle = gzip.open("GenBank/NC_000932.gb.bgz", "r")
>>> assert 0 == handle.tell()
>>> line = handle.readline()
>>> assert 80 == handle.tell()
>>> line = handle.readline()
>>> assert 143 == handle.tell()
>>> data = handle.read(70000)
>>> assert 70143 == handle.tell()
>>> handle.close()

We can also access the file using the BGZF reader - but pay
attention to the file offsets which will be explained below:

>>> handle = BgzfReader("GenBank/NC_000932.gb.bgz", "r")
>>> assert 0 == handle.tell()
>>> print(handle.readline().rstrip())
LOCUS       NC_000932             154478 bp    DNA     circular PLN 15-APR-2009
>>> assert 80 == handle.tell()
>>> print(handle.readline().rstrip())
DEFINITION  Arabidopsis thaliana chloroplast, complete genome.
>>> assert 143 == handle.tell()
>>> data = handle.read(70000)
>>> assert 987828735 == handle.tell()
>>> print(handle.readline().rstrip())
f="GeneID:844718"
>>> print(handle.readline().rstrip())
     CDS             complement(join(84337..84771,85454..85843))
>>> offset = handle.seek(make_virtual_offset(55074, 126))
>>> print(handle.readline().rstrip())
    68521 tatgtcattc gaaattgtat aaagacaact cctatttaat agagctattt gtgcaagtat
>>> handle.close()

Notice the handle's offset looks different as a BGZF file. This
brings us to the key point about BGZF, which is the block structure:

>>> handle = open("GenBank/NC_000932.gb.bgz", "rb")
>>> for values in BgzfBlocks(handle):
...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)
Raw start 0, raw length 15073; data start 0, data length 65536
Raw start 15073, raw length 17857; data start 65536, data length 65536
Raw start 32930, raw length 22144; data start 131072, data length 65536
Raw start 55074, raw length 22230; data start 196608, data length 65536
Raw start 77304, raw length 14939; data start 262144, data length 43478
Raw start 92243, raw length 28; data start 305622, data length 0
>>> handle.close()

In this example the first three blocks are 'full' and hold 65536 bytes
of uncompressed data. The fourth block isn't full and holds 43478 bytes.
Finally there is a special empty fifth block which takes 28 bytes on
disk and serves as an 'end of file' (EOF) marker. If this is missing,
it is possible your BGZF file is incomplete.

By reading ahead 70,000 bytes we moved into the second BGZF block,
and at that point the BGZF virtual offsets start to look different
to a simple offset into the decompressed data as exposed by the gzip
library.

As an example, consider seeking to the decompressed position 196734.
Since 196734 = 65536 + 65536 + 65536 + 126 = 65536*3 + 126, this
is equivalent to jumping the first three blocks (which in this
specific example are all size 65536 after decompression - which
does not always hold) and starting at byte 126 of the fourth block
(after decompression). For BGZF, we need to know the fourth block's
offset of 55074 and the offset within the block of 126 to get the
BGZF virtual offset.

>>> print(55074 << 16 | 126)
3609329790
>>> print(bgzf.make_virtual_offset(55074, 126))
3609329790

Thus for this BGZF file, decompressed position 196734 corresponds
to the virtual offset 3609329790. However, another BGZF file with
different contents would have compressed more or less efficiently,
so the compressed blocks would be different sizes. What this means
is the mapping between the uncompressed offset and the compressed
virtual offset depends on the BGZF file you are using.

If you are accessing a BGZF file via this module, just use the
handle.tell() method to note the virtual offset of a position you
may later want to return to using handle.seek().

The catch with BGZF virtual offsets is while they can be compared
(which offset comes first in the file), you cannot safely subtract
them to get the size of the data between them, nor add/subtract
a relative offset.

Of course you can parse this file with Bio.SeqIO using BgzfReader,
although there isn't any benefit over using gzip.open(...), unless
you want to index BGZF compressed sequence files:

>>> from Bio import SeqIO
>>> handle = BgzfReader("GenBank/NC_000932.gb.bgz")
>>> record = SeqIO.read(handle, "genbank")
>>> handle.close()
>>> print(record.id)
NC_000932.1

Text Mode
---------

Like the standard library gzip.open(...), the BGZF code defaults to opening
files in binary mode.

You can request the file be opened in text mode, but beware that this is hard
coded to the simple "latin1" (aka "iso-8859-1") encoding (which includes all
the ASCII characters), which works well with most Western European languages.
However, it is not fully compatible with the more widely used UTF-8 encoding.

In variable width encodings like UTF-8, some single characters in the unicode
text output are represented by multiple bytes in the raw binary form. This is
problematic with BGZF, as we cannot always decode each block in isolation - a
single unicode character could be split over two blocks. This can even happen
with fixed width unicode encodings, as the BGZF block size is not fixed.

Therefore, this module is currently restricted to only support single byte
unicode encodings, such as ASCII, "latin1" (which is a superset of ASCII), or
potentially other character maps (not implemented).

Furthermore, unlike the default text mode on Python 3, we do not attempt to
implement universal new line mode. This transforms the various operating system
new line conventions like Windows (CR LF or "\\r\\n"), Unix (just LF, "\\n"), or
old Macs (just CR, "\\r"), into just LF ("\\n"). Here we have the same problem -
is "\\r" at the end of a block an incomplete Windows style new line?

Instead, you will get the CR ("\\r") and LF ("\\n") characters as is.

If your data is in UTF-8 or any other incompatible encoding, you must use
binary mode, and decode the appropriate fragments yourself.
"""
import struct
import sys
import zlib
from builtins import open as _open
_bgzf_magic = b'\x1f\x8b\x08\x04'
_bgzf_header = b'\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00'
_bgzf_eof = b'\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00\x1b\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00'
_bytes_BC = b'BC'

def open(filename, mode='rb'):
    if False:
        return 10
    'Open a BGZF file for reading, writing or appending.\n\n    If text mode is requested, in order to avoid multi-byte characters, this is\n    hard coded to use the "latin1" encoding, and "\\r" and "\\n" are passed as is\n    (without implementing universal new line mode).\n\n    If your data is in UTF-8 or any other incompatible encoding, you must use\n    binary mode, and decode the appropriate fragments yourself.\n    '
    if 'r' in mode.lower():
        return BgzfReader(filename, mode)
    elif 'w' in mode.lower() or 'a' in mode.lower():
        return BgzfWriter(filename, mode)
    else:
        raise ValueError(f'Bad mode {mode!r}')

def make_virtual_offset(block_start_offset, within_block_offset):
    if False:
        while True:
            i = 10
    "Compute a BGZF virtual offset from block start and within block offsets.\n\n    The BAM indexing scheme records read positions using a 64 bit\n    'virtual offset', comprising in C terms:\n\n    block_start_offset << 16 | within_block_offset\n\n    Here block_start_offset is the file offset of the BGZF block\n    start (unsigned integer using up to 64-16 = 48 bits), and\n    within_block_offset within the (decompressed) block (unsigned\n    16 bit integer).\n\n    >>> make_virtual_offset(0, 0)\n    0\n    >>> make_virtual_offset(0, 1)\n    1\n    >>> make_virtual_offset(0, 2**16 - 1)\n    65535\n    >>> make_virtual_offset(0, 2**16)\n    Traceback (most recent call last):\n    ...\n    ValueError: Require 0 <= within_block_offset < 2**16, got 65536\n\n    >>> 65536 == make_virtual_offset(1, 0)\n    True\n    >>> 65537 == make_virtual_offset(1, 1)\n    True\n    >>> 131071 == make_virtual_offset(1, 2**16 - 1)\n    True\n\n    >>> 6553600000 == make_virtual_offset(100000, 0)\n    True\n    >>> 6553600001 == make_virtual_offset(100000, 1)\n    True\n    >>> 6553600010 == make_virtual_offset(100000, 10)\n    True\n\n    >>> make_virtual_offset(2**48, 0)\n    Traceback (most recent call last):\n    ...\n    ValueError: Require 0 <= block_start_offset < 2**48, got 281474976710656\n\n    "
    if within_block_offset < 0 or within_block_offset >= 65536:
        raise ValueError('Require 0 <= within_block_offset < 2**16, got %i' % within_block_offset)
    if block_start_offset < 0 or block_start_offset >= 281474976710656:
        raise ValueError('Require 0 <= block_start_offset < 2**48, got %i' % block_start_offset)
    return block_start_offset << 16 | within_block_offset

def split_virtual_offset(virtual_offset):
    if False:
        i = 10
        return i + 15
    'Divides a 64-bit BGZF virtual offset into block start & within block offsets.\n\n    >>> (100000, 0) == split_virtual_offset(6553600000)\n    True\n    >>> (100000, 10) == split_virtual_offset(6553600010)\n    True\n\n    '
    start = virtual_offset >> 16
    return (start, virtual_offset ^ start << 16)

def BgzfBlocks(handle):
    if False:
        print('Hello World!')
    'Low level debugging function to inspect BGZF blocks.\n\n    Expects a BGZF compressed file opened in binary read mode using\n    the builtin open function. Do not use a handle from this bgzf\n    module or the gzip module\'s open function which will decompress\n    the file.\n\n    Returns the block start offset (see virtual offsets), the block\n    length (add these for the start of the next block), and the\n    decompressed length of the blocks contents (limited to 65536 in\n    BGZF), as an iterator - one tuple per BGZF block.\n\n    >>> from builtins import open\n    >>> handle = open("SamBam/ex1.bam", "rb")\n    >>> for values in BgzfBlocks(handle):\n    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)\n    Raw start 0, raw length 18239; data start 0, data length 65536\n    Raw start 18239, raw length 18223; data start 65536, data length 65536\n    Raw start 36462, raw length 18017; data start 131072, data length 65536\n    Raw start 54479, raw length 17342; data start 196608, data length 65536\n    Raw start 71821, raw length 17715; data start 262144, data length 65536\n    Raw start 89536, raw length 17728; data start 327680, data length 65536\n    Raw start 107264, raw length 17292; data start 393216, data length 63398\n    Raw start 124556, raw length 28; data start 456614, data length 0\n    >>> handle.close()\n\n    Indirectly we can tell this file came from an old version of\n    samtools because all the blocks (except the final one and the\n    dummy empty EOF marker block) are 65536 bytes.  Later versions\n    avoid splitting a read between two blocks, and give the header\n    its own block (useful to speed up replacing the header). You\n    can see this in ex1_refresh.bam created using samtools 0.1.18:\n\n    samtools view -b ex1.bam > ex1_refresh.bam\n\n    >>> handle = open("SamBam/ex1_refresh.bam", "rb")\n    >>> for values in BgzfBlocks(handle):\n    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)\n    Raw start 0, raw length 53; data start 0, data length 38\n    Raw start 53, raw length 18195; data start 38, data length 65434\n    Raw start 18248, raw length 18190; data start 65472, data length 65409\n    Raw start 36438, raw length 18004; data start 130881, data length 65483\n    Raw start 54442, raw length 17353; data start 196364, data length 65519\n    Raw start 71795, raw length 17708; data start 261883, data length 65411\n    Raw start 89503, raw length 17709; data start 327294, data length 65466\n    Raw start 107212, raw length 17390; data start 392760, data length 63854\n    Raw start 124602, raw length 28; data start 456614, data length 0\n    >>> handle.close()\n\n    The above example has no embedded SAM header (thus the first block\n    is very small at just 38 bytes of decompressed data), while the next\n    example does (a larger block of 103 bytes). Notice that the rest of\n    the blocks show the same sizes (they contain the same read data):\n\n    >>> handle = open("SamBam/ex1_header.bam", "rb")\n    >>> for values in BgzfBlocks(handle):\n    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)\n    Raw start 0, raw length 104; data start 0, data length 103\n    Raw start 104, raw length 18195; data start 103, data length 65434\n    Raw start 18299, raw length 18190; data start 65537, data length 65409\n    Raw start 36489, raw length 18004; data start 130946, data length 65483\n    Raw start 54493, raw length 17353; data start 196429, data length 65519\n    Raw start 71846, raw length 17708; data start 261948, data length 65411\n    Raw start 89554, raw length 17709; data start 327359, data length 65466\n    Raw start 107263, raw length 17390; data start 392825, data length 63854\n    Raw start 124653, raw length 28; data start 456679, data length 0\n    >>> handle.close()\n\n    '
    if isinstance(handle, BgzfReader):
        raise TypeError('Function BgzfBlocks expects a binary handle')
    data_start = 0
    while True:
        start_offset = handle.tell()
        try:
            (block_length, data) = _load_bgzf_block(handle)
        except StopIteration:
            break
        data_len = len(data)
        yield (start_offset, block_length, data_start, data_len)
        data_start += data_len

def _load_bgzf_block(handle, text_mode=False):
    if False:
        i = 10
        return i + 15
    'Load the next BGZF block of compressed data (PRIVATE).\n\n    Returns a tuple (block size and data), or at end of file\n    will raise StopIteration.\n    '
    magic = handle.read(4)
    if not magic:
        raise StopIteration
    if magic != _bgzf_magic:
        raise ValueError('A BGZF (e.g. a BAM file) block should start with %r, not %r; handle.tell() now says %r' % (_bgzf_magic, magic, handle.tell()))
    (gzip_mod_time, gzip_extra_flags, gzip_os, extra_len) = struct.unpack('<LBBH', handle.read(8))
    block_size = None
    x_len = 0
    while x_len < extra_len:
        subfield_id = handle.read(2)
        subfield_len = struct.unpack('<H', handle.read(2))[0]
        subfield_data = handle.read(subfield_len)
        x_len += subfield_len + 4
        if subfield_id == _bytes_BC:
            if subfield_len != 2:
                raise ValueError('Wrong BC payload length')
            if block_size is not None:
                raise ValueError('Two BC subfields?')
            block_size = struct.unpack('<H', subfield_data)[0] + 1
    if x_len != extra_len:
        raise ValueError(f'x_len and extra_len differ {x_len}, {extra_len}')
    if block_size is None:
        raise ValueError("Missing BC, this isn't a BGZF file!")
    deflate_size = block_size - 1 - extra_len - 19
    d = zlib.decompressobj(-15)
    data = d.decompress(handle.read(deflate_size)) + d.flush()
    expected_crc = handle.read(4)
    expected_size = struct.unpack('<I', handle.read(4))[0]
    if expected_size != len(data):
        raise RuntimeError('Decompressed to %i, not %i' % (len(data), expected_size))
    crc = zlib.crc32(data)
    if crc < 0:
        crc = struct.pack('<i', crc)
    else:
        crc = struct.pack('<I', crc)
    if expected_crc != crc:
        raise RuntimeError(f'CRC is {crc}, not {expected_crc}')
    if text_mode:
        return (block_size, data.decode('latin-1'))
    else:
        return (block_size, data)

class BgzfReader:
    """BGZF reader, acts like a read only handle but seek/tell differ.

    Let's use the BgzfBlocks function to have a peek at the BGZF blocks
    in an example BAM file,

    >>> from builtins import open
    >>> handle = open("SamBam/ex1.bam", "rb")
    >>> for values in BgzfBlocks(handle):
    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)
    Raw start 0, raw length 18239; data start 0, data length 65536
    Raw start 18239, raw length 18223; data start 65536, data length 65536
    Raw start 36462, raw length 18017; data start 131072, data length 65536
    Raw start 54479, raw length 17342; data start 196608, data length 65536
    Raw start 71821, raw length 17715; data start 262144, data length 65536
    Raw start 89536, raw length 17728; data start 327680, data length 65536
    Raw start 107264, raw length 17292; data start 393216, data length 63398
    Raw start 124556, raw length 28; data start 456614, data length 0
    >>> handle.close()

    Now let's see how to use this block information to jump to
    specific parts of the decompressed BAM file:

    >>> handle = BgzfReader("SamBam/ex1.bam", "rb")
    >>> assert 0 == handle.tell()
    >>> magic = handle.read(4)
    >>> assert 4 == handle.tell()

    So far nothing so strange, we got the magic marker used at the
    start of a decompressed BAM file, and the handle position makes
    sense. Now however, let's jump to the end of this block and 4
    bytes into the next block by reading 65536 bytes,

    >>> data = handle.read(65536)
    >>> len(data)
    65536
    >>> assert 1195311108 == handle.tell()

    Expecting 4 + 65536 = 65540 were you? Well this is a BGZF 64-bit
    virtual offset, which means:

    >>> split_virtual_offset(1195311108)
    (18239, 4)

    You should spot 18239 as the start of the second BGZF block, while
    the 4 is the offset into this block. See also make_virtual_offset,

    >>> make_virtual_offset(18239, 4)
    1195311108

    Let's jump back to almost the start of the file,

    >>> make_virtual_offset(0, 2)
    2
    >>> handle.seek(2)
    2
    >>> handle.close()

    Note that you can use the max_cache argument to limit the number of
    BGZF blocks cached in memory. The default is 100, and since each
    block can be up to 64kb, the default cache could take up to 6MB of
    RAM. The cache is not important for reading through the file in one
    pass, but is important for improving performance of random access.
    """

    def __init__(self, filename=None, mode='r', fileobj=None, max_cache=100):
        if False:
            return 10
        'Initialize the class for reading a BGZF file.\n\n        You would typically use the top level ``bgzf.open(...)`` function\n        which will call this class internally. Direct use is discouraged.\n\n        Either the ``filename`` (string) or ``fileobj`` (input file object in\n        binary mode) arguments must be supplied, but not both.\n\n        Argument ``mode`` controls if the data will be returned as strings in\n        text mode ("rt", "tr", or default "r"), or bytes binary mode ("rb"\n        or "br"). The argument name matches the built-in ``open(...)`` and\n        standard library ``gzip.open(...)`` function.\n\n        If text mode is requested, in order to avoid multi-byte characters,\n        this is hard coded to use the "latin1" encoding, and "\\r" and "\\n"\n        are passed as is (without implementing universal new line mode). There\n        is no ``encoding`` argument.\n\n        If your data is in UTF-8 or any other incompatible encoding, you must\n        use binary mode, and decode the appropriate fragments yourself.\n\n        Argument ``max_cache`` controls the maximum number of BGZF blocks to\n        cache in memory. Each can be up to 64kb thus the default of 100 blocks\n        could take up to 6MB of RAM. This is important for efficient random\n        access, a small value is fine for reading the file in one pass.\n        '
        if max_cache < 1:
            raise ValueError('Use max_cache with a minimum of 1')
        if filename and fileobj:
            raise ValueError('Supply either filename or fileobj, not both')
        if mode.lower() not in ('r', 'tr', 'rt', 'rb', 'br'):
            raise ValueError("Must use a read mode like 'r' (default), 'rt', or 'rb' for binary")
        if fileobj:
            if fileobj.read(0) != b'':
                raise ValueError('fileobj not opened in binary mode')
            handle = fileobj
        else:
            handle = _open(filename, 'rb')
        self._text = 'b' not in mode.lower()
        if self._text:
            self._newline = '\n'
        else:
            self._newline = b'\n'
        self._handle = handle
        self.max_cache = max_cache
        self._buffers = {}
        self._block_start_offset = None
        self._block_raw_length = None
        self._load_block(handle.tell())

    def _load_block(self, start_offset=None):
        if False:
            print('Hello World!')
        if start_offset is None:
            start_offset = self._block_start_offset + self._block_raw_length
        if start_offset == self._block_start_offset:
            self._within_block_offset = 0
            return
        elif start_offset in self._buffers:
            (self._buffer, self._block_raw_length) = self._buffers[start_offset]
            self._within_block_offset = 0
            self._block_start_offset = start_offset
            return
        while len(self._buffers) >= self.max_cache:
            self._buffers.popitem()
        handle = self._handle
        if start_offset is not None:
            handle.seek(start_offset)
        self._block_start_offset = handle.tell()
        try:
            (block_size, self._buffer) = _load_bgzf_block(handle, self._text)
        except StopIteration:
            block_size = 0
            if self._text:
                self._buffer = ''
            else:
                self._buffer = b''
        self._within_block_offset = 0
        self._block_raw_length = block_size
        self._buffers[self._block_start_offset] = (self._buffer, block_size)

    def tell(self):
        if False:
            print('Hello World!')
        'Return a 64-bit unsigned BGZF virtual offset.'
        if 0 < self._within_block_offset and self._within_block_offset == len(self._buffer):
            return self._block_start_offset + self._block_raw_length << 16
        else:
            return self._block_start_offset << 16 | self._within_block_offset

    def seek(self, virtual_offset):
        if False:
            print('Hello World!')
        'Seek to a 64-bit unsigned BGZF virtual offset.'
        start_offset = virtual_offset >> 16
        within_block = virtual_offset ^ start_offset << 16
        if start_offset != self._block_start_offset:
            self._load_block(start_offset)
            if start_offset != self._block_start_offset:
                raise ValueError('start_offset not loaded correctly')
        if within_block > len(self._buffer):
            if not (within_block == 0 and len(self._buffer) == 0):
                raise ValueError('Within offset %i but block size only %i' % (within_block, len(self._buffer)))
        self._within_block_offset = within_block
        return virtual_offset

    def read(self, size=-1):
        if False:
            while True:
                i = 10
        'Read method for the BGZF module.'
        if size < 0:
            raise NotImplementedError("Don't be greedy, that could be massive!")
        result = '' if self._text else b''
        while size and self._block_raw_length:
            if self._within_block_offset + size <= len(self._buffer):
                data = self._buffer[self._within_block_offset:self._within_block_offset + size]
                self._within_block_offset += size
                if not data:
                    raise ValueError('Must be at least 1 byte')
                result += data
                break
            else:
                data = self._buffer[self._within_block_offset:]
                size -= len(data)
                self._load_block()
                result += data
        return result

    def readline(self):
        if False:
            return 10
        'Read a single line for the BGZF file.'
        result = '' if self._text else b''
        while self._block_raw_length:
            i = self._buffer.find(self._newline, self._within_block_offset)
            if i == -1:
                data = self._buffer[self._within_block_offset:]
                self._load_block()
                result += data
            elif i + 1 == len(self._buffer):
                data = self._buffer[self._within_block_offset:]
                self._load_block()
                if not data:
                    raise ValueError('Must be at least 1 byte')
                result += data
                break
            else:
                data = self._buffer[self._within_block_offset:i + 1]
                self._within_block_offset = i + 1
                result += data
                break
        return result

    def __next__(self):
        if False:
            return 10
        'Return the next line.'
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __iter__(self):
        if False:
            return 10
        'Iterate over the lines in the BGZF file.'
        return self

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        'Close BGZF file.'
        self._handle.close()
        self._buffer = None
        self._block_start_offset = None
        self._buffers = None

    def seekable(self):
        if False:
            print('Hello World!')
        'Return True indicating the BGZF supports random access.'
        return True

    def isatty(self):
        if False:
            return 10
        'Return True if connected to a TTY device.'
        return False

    def fileno(self):
        if False:
            while True:
                i = 10
        'Return integer file descriptor.'
        return self._handle.fileno()

    def __enter__(self):
        if False:
            while True:
                i = 10
        'Open a file operable with WITH statement.'
        return self

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        'Close a file with WITH statement.'
        self.close()

class BgzfWriter:
    """Define a BGZFWriter object."""

    def __init__(self, filename=None, mode='w', fileobj=None, compresslevel=6):
        if False:
            i = 10
            return i + 15
        'Initilize the class.'
        if filename and fileobj:
            raise ValueError('Supply either filename or fileobj, not both')
        if fileobj:
            if fileobj.read(0) != b'':
                raise ValueError('fileobj not opened in binary mode')
            handle = fileobj
        else:
            if 'w' not in mode.lower() and 'a' not in mode.lower():
                raise ValueError(f'Must use write or append mode, not {mode!r}')
            if 'a' in mode.lower():
                handle = _open(filename, 'ab')
            else:
                handle = _open(filename, 'wb')
        self._text = 'b' not in mode.lower()
        self._handle = handle
        self._buffer = b''
        self.compresslevel = compresslevel

    def _write_block(self, block):
        if False:
            i = 10
            return i + 15
        'Write provided data to file as a single BGZF compressed block (PRIVATE).'
        if len(block) > 65536:
            raise ValueError(f'{len(block)} Block length > 65536')
        c = zlib.compressobj(self.compresslevel, zlib.DEFLATED, -15, zlib.DEF_MEM_LEVEL, 0)
        compressed = c.compress(block) + c.flush()
        del c
        if len(compressed) > 65536:
            raise RuntimeError("TODO - Didn't compress enough, try less data in this block")
        crc = zlib.crc32(block)
        if crc < 0:
            crc = struct.pack('<i', crc)
        else:
            crc = struct.pack('<I', crc)
        bsize = struct.pack('<H', len(compressed) + 25)
        crc = struct.pack('<I', zlib.crc32(block) & 4294967295)
        uncompressed_length = struct.pack('<I', len(block))
        data = _bgzf_header + bsize + compressed + crc + uncompressed_length
        self._handle.write(data)

    def write(self, data):
        if False:
            while True:
                i = 10
        'Write method for the class.'
        if isinstance(data, str):
            data = data.encode('latin-1')
        data_len = len(data)
        if len(self._buffer) + data_len < 65536:
            self._buffer += data
        else:
            self._buffer += data
            while len(self._buffer) >= 65536:
                self._write_block(self._buffer[:65536])
                self._buffer = self._buffer[65536:]

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        'Flush data explicitally.'
        while len(self._buffer) >= 65536:
            self._write_block(self._buffer[:65535])
            self._buffer = self._buffer[65535:]
        self._write_block(self._buffer)
        self._buffer = b''
        self._handle.flush()

    def close(self):
        if False:
            print('Hello World!')
        'Flush data, write 28 bytes BGZF EOF marker, and close BGZF file.\n\n        samtools will look for a magic EOF marker, just a 28 byte empty BGZF\n        block, and if it is missing warns the BAM file may be truncated. In\n        addition to samtools writing this block, so too does bgzip - so this\n        implementation does too.\n        '
        if self._buffer:
            self.flush()
        self._handle.write(_bgzf_eof)
        self._handle.flush()
        self._handle.close()

    def tell(self):
        if False:
            print('Hello World!')
        'Return a BGZF 64-bit virtual offset.'
        return make_virtual_offset(self._handle.tell(), len(self._buffer))

    def seekable(self):
        if False:
            i = 10
            return i + 15
        'Return True indicating the BGZF supports random access.'
        return False

    def isatty(self):
        if False:
            return 10
        'Return True if connected to a TTY device.'
        return False

    def fileno(self):
        if False:
            print('Hello World!')
        'Return integer file descriptor.'
        return self._handle.fileno()

    def __enter__(self):
        if False:
            return 10
        'Open a file operable with WITH statement.'
        return self

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        'Close a file with WITH statement.'
        self.close()
if __name__ == '__main__':
    if len(sys.argv) > 1:
        print('Call this with no arguments and pipe uncompressed data in on stdin')
        print('and it will produce BGZF compressed data on stdout. e.g.')
        print('')
        print('./bgzf.py < example.fastq > example.fastq.bgz')
        print('')
        print('The extension convention of *.bgz is to distinugish these from *.gz')
        print('used for standard gzipped files without the block structure of BGZF.')
        print('You can use the standard gunzip command to decompress BGZF files,')
        print('if it complains about the extension try something like this:')
        print('')
        print('cat example.fastq.bgz | gunzip > example.fastq')
        print('')
        print('See also the tool bgzip that comes with samtools')
        sys.exit(0)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    sys.stderr.write('Producing BGZF output from stdin...\n')
    w = BgzfWriter(fileobj=stdout)
    while True:
        data = stdin.read(65536)
        w.write(data)
        if not data:
            break
    w.close()
    sys.stderr.write('BGZF data produced\n')