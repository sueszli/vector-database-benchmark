"""Bio.SeqIO support for UCSC's "twoBit" (.2bit) file format.

This parser reads the index stored in the twoBit file, as well as the masked
regions and the N's for each sequence. It also creates sequence data objects
(_TwoBitSequenceData objects), which support only two methods: __len__ and
__getitem__. The former will return the length of the sequence, while the
latter returns the sequence (as a bytes object) for the requested region.

Using the information in the index, the __getitem__ method calculates the file
position at which the requested region starts, and only reads the requested
sequence region. Note that the full sequence of a record is loaded only if
specifically requested, making the parser memory-efficient.

The TwoBitIterator object implements the __getitem__, keys, and __len__
methods that allow it to be used as a dictionary.
"""
try:
    import numpy as np
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install NumPy if you want to use Bio.SeqIO with TwoBit files.See http://www.numpy.org/') from None
from Bio.Seq import Seq
from Bio.Seq import SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord
from . import _twoBitIO
from .Interfaces import SequenceIterator

class _TwoBitSequenceData(SequenceDataAbstractBaseClass):
    """Stores information needed to retrieve sequence data from a .2bit file (PRIVATE).

    Objects of this class store the file position at which the sequence data
    start, the sequence length, and the start and end position of unknown (N)
    and masked (lowercase) letters in the sequence.

    Only two methods are provided: __len__ and __getitem__. The former will
    return the length of the sequence, while the latter returns the sequence
    (as a bytes object) for the requested region. The full sequence of a record
    is loaded only if explicitly requested.
    """
    __slots__ = ('stream', 'offset', 'length', 'nBlocks', 'maskBlocks')

    def __init__(self, stream, offset, length):
        if False:
            print('Hello World!')
        'Initialize the file stream and file position of the sequence data.'
        self.stream = stream
        self.offset = offset
        self.length = length
        super().__init__()

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        'Return the sequence contents (as a bytes object) for the requested region.'
        length = self.length
        if isinstance(key, slice):
            (start, end, step) = key.indices(length)
            size = len(range(start, end, step))
            if size == 0:
                return b''
        else:
            if key < 0:
                key += length
                if key < 0:
                    raise IndexError('index out of range')
            start = key
            end = key + 1
            step = 1
            size = 1
        byteStart = start // 4
        byteEnd = (end + 3) // 4
        byteSize = byteEnd - byteStart
        stream = self.stream
        try:
            stream.seek(self.offset + byteStart)
        except ValueError as exception:
            if str(exception) == 'seek of closed file':
                raise ValueError('cannot retrieve sequence: file is closed') from None
            raise
        data = np.fromfile(stream, dtype='uint8', count=byteSize)
        sequence = _twoBitIO.convert(data, start, end, step, self.nBlocks, self.maskBlocks)
        if isinstance(key, slice):
            return sequence
        else:
            return ord(sequence)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the sequence length.'
        return self.length

    def upper(self):
        if False:
            i = 10
            return i + 15
        'Remove the sequence mask.'
        data = _TwoBitSequenceData(self.stream, self.offset, self.length)
        data.nBlocks = self.nBlocks[:, :]
        data.maskBlocks = np.empty((0, 2), dtype='uint32')
        return data

    def lower(self):
        if False:
            for i in range(10):
                print('nop')
        'Extend the sequence mask to the full sequence.'
        data = _TwoBitSequenceData(self.stream, self.offset, self.length)
        data.nBlocks = self.nBlocks[:, :]
        data.maskBlocks = np.array([[0, self.length]], dtype='uint32')
        return data

class TwoBitIterator(SequenceIterator):
    """Parser for UCSC twoBit (.2bit) files."""

    def __init__(self, source):
        if False:
            return 10
        'Read the file index.'
        super().__init__(source, mode='b', fmt='twoBit')
        self.should_close_stream = False
        stream = self.stream
        data = stream.read(4)
        if not data:
            raise ValueError('Empty file.')
        byteorders = ('little', 'big')
        dtypes = ('<u4', '>u4')
        for (byteorder, dtype) in zip(byteorders, dtypes):
            signature = int.from_bytes(data, byteorder)
            if signature == 440477507:
                break
        else:
            raise ValueError('Unknown signature')
        self.byteorder = byteorder
        data = stream.read(4)
        version = int.from_bytes(data, byteorder, signed=False)
        if version == 1:
            raise ValueError('version-1 twoBit files with 64-bit offsets for index are currently not supported')
        if version != 0:
            raise ValueError('Found unexpected file version %u; aborting' % version)
        data = stream.read(4)
        sequenceCount = int.from_bytes(data, byteorder, signed=False)
        data = stream.read(4)
        reserved = int.from_bytes(data, byteorder, signed=False)
        if reserved != 0:
            raise ValueError('Found non-zero reserved field; aborting')
        sequences = {}
        for i in range(sequenceCount):
            data = stream.read(1)
            nameSize = int.from_bytes(data, byteorder, signed=False)
            data = stream.read(nameSize)
            name = data.decode('ASCII')
            data = stream.read(4)
            offset = int.from_bytes(data, byteorder, signed=False)
            sequences[name] = (stream, offset)
        self.sequences = sequences
        for (name, (stream, offset)) in sequences.items():
            stream.seek(offset)
            data = stream.read(4)
            dnaSize = int.from_bytes(data, byteorder, signed=False)
            sequence = _TwoBitSequenceData(stream, offset, dnaSize)
            data = stream.read(4)
            nBlockCount = int.from_bytes(data, byteorder, signed=False)
            nBlockStarts = np.fromfile(stream, dtype=dtype, count=nBlockCount)
            nBlockSizes = np.fromfile(stream, dtype=dtype, count=nBlockCount)
            sequence.nBlocks = np.empty((nBlockCount, 2), dtype='uint32')
            sequence.nBlocks[:, 0] = nBlockStarts
            sequence.nBlocks[:, 1] = nBlockStarts + nBlockSizes
            data = stream.read(4)
            maskBlockCount = int.from_bytes(data, byteorder, signed=False)
            maskBlockStarts = np.fromfile(stream, dtype=dtype, count=maskBlockCount)
            maskBlockSizes = np.fromfile(stream, dtype=dtype, count=maskBlockCount)
            sequence.maskBlocks = np.empty((maskBlockCount, 2), dtype='uint32')
            sequence.maskBlocks[:, 0] = maskBlockStarts
            sequence.maskBlocks[:, 1] = maskBlockStarts + maskBlockSizes
            data = stream.read(4)
            reserved = int.from_bytes(data, byteorder, signed=False)
            if reserved != 0:
                raise ValueError('Found non-zero reserved field %u' % reserved)
            sequence.offset = stream.tell()
            sequences[name] = sequence

    def parse(self, stream):
        if False:
            i = 10
            return i + 15
        'Iterate over the sequences in the file.'
        for (name, sequence) in self.sequences.items():
            sequence = Seq(sequence)
            record = SeqRecord(sequence, id=name)
            yield record

    def __getitem__(self, name):
        if False:
            while True:
                i = 10
        'Return sequence associated with given name as a SeqRecord object.'
        try:
            sequence = self.sequences[name]
        except ValueError:
            raise KeyError(name) from None
        sequence = Seq(sequence)
        return SeqRecord(sequence, id=name)

    def keys(self):
        if False:
            return 10
        'Return a list with the names of the sequences in the file.'
        return self.sequences.keys()

    def __len__(self):
        if False:
            return 10
        'Return number of sequences.'
        return len(self.sequences)