"""
Provides CABFile, an extractor for the MSCAB format.

Struct definitions are according to the documentation at
https://msdn.microsoft.com/en-us/library/bb417343.aspx.
"""
from __future__ import annotations
import typing
from bisect import bisect
from calendar import timegm
from collections import OrderedDict
from typing import Generator, NoReturn, Union
from ..log import dbg
from ..util.filelike.readonly import PosSavingReadOnlyFileLikeObject
from ..util.filelike.stream import StreamFragment
from ..util.files import read_guaranteed, read_nullterminated_string
from ..util.fslike.filecollection import FileCollection
from ..util.math import INF
from ..util.strings import try_decode
from ..util.struct import NamedStruct, Flags
from .cabchecksum import mscab_csum
if typing.TYPE_CHECKING:
    from openage.util.filelike.abstract import FileLikeObject

class CFHeaderFlags(Flags):
    """ Cabinet file option indicators. Found in the header. """
    specstr = 'H'
    prev_cabinet = 0
    next_cabinet = 1
    reserve_present = 2

class CFHeader(NamedStruct):
    """ Global CAB file header; found at the very beginning of the file. """
    endianness = '<'
    signature = '4s'
    reserved1 = 'I'
    cbCabinet = 'I'
    reserved2 = 'I'
    coffFiles = 'I'
    reserved3 = 'I'
    versionMinor = 'B'
    versionMajor = 'B'
    cFolders = 'H'
    cFiles = 'H'
    flags = CFHeaderFlags
    setID = 'H'
    iCabinet = 'H'
    reserved_data = None
    reserved = None
    prev_cab = None
    prev_disk = None
    next_cab = None
    next_disk = None

class CFHeaderReservedFields(NamedStruct):
    """
    Optionally found after the header.
    The fields indicate the size of the reserved data blocks in the header,
    folder and data structs.
    """
    endianness = '<'
    cbCFHeader = 'H'
    cbCFFolder = 'B'
    cbCFData = 'B'

class CFFolder(NamedStruct):
    """
    CAB folder header; A CAB folder is a data stream consisting of
    (compressed) concatenated file contents.
    """
    endianness = '<'
    coffCabStart = 'I'
    cCFData = 'H'
    typeCompress = 'H'
    reserved = None
    comp_name = None
    plain_stream = None

class CFFileAttributes(Flags):
    """
    File flags; found in the CFFile struct.
    """
    specstr = 'H'
    rdonly = 0
    hidden = 1
    system = 2
    arch = 5
    exec = 6
    name_is_utf = 7

class CFFile(NamedStruct):
    """
    Header for a single file.

    Describes the file's metadata,
    as well as the location of its content (which CAB folder, at what offset).
    """
    endianness = '<'
    size = 'I'
    pos = 'I'
    folderid = 'H'
    date = 'H'
    time = 'H'
    attribs = CFFileAttributes
    path = None
    continued = None
    continues = None
    folder = None
    timestamp = None

class CFData(NamedStruct):
    """
    CAB folders are concatenations of data blocks; this is the header
    of one such data block.
    """
    endianness = '<'
    csum = 'I'
    cbData = 'H'
    cbUncomp = 'H'
    reserved = None
    payload = None

    def verify_checksum(self) -> Union[None, NoReturn]:
        if False:
            i = 10
            return i + 15
        '\n        Checks whether csum contains the correct checksum for the block.\n        Raises ValueError otherwise.\n        '
        checksum = self.cbUncomp << 16 | self.cbData
        if self.reserved:
            checksum ^= mscab_csum(self.reserved)
        checksum ^= mscab_csum(self.payload)
        if checksum != self.csum:
            raise ValueError('checksum error in MSCAB data block')

class CABFile(FileCollection):
    """
    The actual file system-like CAB object.

    Constructor arguments:

    @param cab:
        A file-like object that must implement read() and seek() with
        whence=os.SEEK_SET.

    The constructor reads the entire header, including the folder and file
    descriptions. Most CAB file issues should cause the constructor to fail.
    """

    def __init__(self, cab: FileLikeObject, offset: int=0):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        cab.seek(offset)
        header = CFHeader.read(cab)
        if header.signature != b'MSCF':
            raise SyntaxError('invalid CAB file signature: ' + repr(header.signature))
        if header.flags.reserve_present:
            header.reserved_data = CFHeaderReservedFields.read(cab)
        else:
            header.reserved_data = CFHeaderReservedFields.from_nullbytes()
        header.reserved = read_guaranteed(cab, header.reserved_data.cbCFHeader)
        if header.flags.prev_cabinet:
            header.prev_cab = try_decode(read_nullterminated_string(cab))
            header.prev_disk = try_decode(read_nullterminated_string(cab))
        if header.flags.next_cabinet:
            header.next_cab = try_decode(read_nullterminated_string(cab))
            header.next_disk = try_decode(read_nullterminated_string(cab))
        dbg(header)
        self.header = header
        self.folders = tuple(self.read_folder_headers(cab, offset))
        self.rootdir = (OrderedDict(), OrderedDict())
        for fileobj in self.read_file_headers(cab, offset):
            if self.is_file(fileobj.path) or self.is_dir(fileobj.path):
                raise ValueError('CABFile has multiple entries with the same path: ' + b'/'.join(fileobj.path).decode())

            def open_r(fileobj=fileobj):
                if False:
                    for i in range(10):
                        print('nop')
                " Returns a opened ('rb') file-like object for fileobj. "
                return StreamFragment(fileobj.folder.plain_stream, fileobj.pos, fileobj.size)
            self.add_fileentry(fileobj.path, (open_r, None, lambda fileobj=fileobj: fileobj.size, lambda fileobj=fileobj: fileobj.timestamp))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'CABFile'

    def read_folder_headers(self, cab: FileLikeObject, offset: int) -> Generator[CFFolder, None, None]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Called during the constructor run.\n\n        Reads the folder headers and initializes the folder's plain stream\n        file-like objects.\n\n        Yields all folders.\n        "
        for _ in range(self.header.cFolders):
            folder = CFFolder.read(cab)
            folder.reserved = read_guaranteed(cab, self.header.reserved_data.cbCFFolder)
            compressed_data_stream = CABFolderStream(cab, folder.coffCabStart + offset, folder.cCFData, self.header.reserved_data.cbCFData)
            compression_type = folder.typeCompress & 15
            if compression_type == 0:
                folder.comp_name = 'Plain'
                folder.plain_stream = compressed_data_stream
            elif compression_type == 1:
                raise SyntaxError('MSZIP compression is unsupported')
            elif compression_type == 2:
                raise SyntaxError('Quantum compression is unsupported')
            elif compression_type == 3:
                window_bits = folder.typeCompress >> 8 & 31
                folder.comp_name = f'LZX (window_bits = {window_bits:d})'
                from .lzxdstream import LZXDStream
                from ..util.filelike.stream import StreamSeekBuffer
                unseekable_plain_stream = LZXDStream(compressed_data_stream, window_bits=window_bits, reset_interval=0)
                folder.plain_stream = StreamSeekBuffer(unseekable_plain_stream)
            else:
                raise SyntaxError(f'Unknown compression type {compression_type:d}')
            dbg(folder)
            yield folder

    def read_file_headers(self, cab: FileLikeObject, offset: int) -> Generator[CFFile, None, None]:
        if False:
            return 10
        '\n        Called during the constructor run.\n\n        Reads the headers for all files and yields CFFile objects.\n        '
        if cab.tell() != self.header.coffFiles + offset:
            cab.seek(self.header.coffFiles + offset)
            dbg('cabfile has nonstandard format: seek to header.coffFiles was required')
        for _ in range(self.header.cFiles):
            fileobj = CFFile.read(cab)
            rpath = read_nullterminated_string(cab)
            if fileobj.attribs.name_is_utf:
                path = rpath.decode('utf-8')
            else:
                path = rpath.decode('iso-8859-1')
            fileobj.path = path.replace('\\', '/').lower().encode().split(b'/')
            if fileobj.folderid == 65533:
                fileobj.folderid = 0
                fileobj.continued = True
            elif fileobj.folderid == 65534:
                fileobj.folderid = len(self.folders) - 1
                fileobj.continues = True
            elif fileobj.folderid == 65535:
                fileobj.folderid = 0
                fileobj.continued = True
                fileobj.continues = True
            fileobj.folder = self.folders[fileobj.folderid]
            year = (fileobj.date >> 9) + 1980
            month = fileobj.date >> 5 & 15
            day = fileobj.date >> 0 & 31
            hour = fileobj.time >> 11
            minute = fileobj.time >> 5 & 63
            sec = fileobj.time << 1 & 63
            fileobj.timestamp = timegm((year, month, day, hour, minute, sec))
            yield fileobj

class CABFolderStream(PosSavingReadOnlyFileLikeObject):
    """
    Read-only, seekable, file-like stream object that represents
    a compressed MSCAB folder (MSCAB folders are just compressed streams
    of concatenated file contents, and are not to be confused with file system
    folders).

    Constructor arguments:

    @param fileobj:
        Seekable file-like object that represents the CAB file.

        CABFolderStream explicitly positions the file cursor before every read;
        this is to make sure that multiple CABFolderStreams can work on the
        same file, in parallel.

        The file object must implement read() and seek() with SEEK_SET.

    @param offset:
        Offset of the first of the folder's data blocks in the CAB file.

    @param blockcount:
        Number of data blocks in the folder.
    """

    def __init__(self, fileobj: FileLikeObject, offset: int, blockcount: int, blockreserved: int):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fileobj = fileobj
        self.blockcount = blockcount
        self.blockreserved = blockreserved
        self.blockoffsets = [offset]
        self.streamindex = [0]

    def next_block_size(self, payloadsize: int) -> None:
        if False:
            print('Hello World!')
        '\n        adds metadata for the next block\n        '
        self.blockoffsets.append(self.blockoffsets[-1] + CFData.size() + self.blockreserved + payloadsize)
        self.streamindex.append(self.streamindex[-1] + payloadsize)

    def read_block_data(self, block_id: int) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        '\n        reads the data of block block_id.\n\n        if necessary, the metadata info in self.blockvalues and\n        self.blockoffsets is updated.\n\n        returns the block data.\n        '
        if block_id >= self.blockcount:
            raise EOFError()
        while block_id >= len(self.blockoffsets):
            offset = self.blockoffsets[-1]
            self.fileobj.seek(self.blockoffsets[-1])
            datablock = CFData.read(self.fileobj)
            self.next_block_size(datablock.cbData)
        offset = self.blockoffsets[block_id]
        self.fileobj.seek(offset)
        datablock = CFData.read(self.fileobj)
        datablock.reserved = read_guaranteed(self.fileobj, self.blockreserved)
        datablock.payload = read_guaranteed(self.fileobj, datablock.cbData)
        datablock.verify_checksum()
        if block_id + 1 == len(self.blockoffsets):
            self.next_block_size(datablock.cbData)
        return datablock.payload

    def read_blocks(self, size: int=-1) -> Generator[bytes, None, None]:
        if False:
            while True:
                i = 10
        '\n        Similar to read, bit instead of a single bytes object,\n        returns an iterator of multiple bytes objects, one for each block.\n\n        Used internally be read(), but you may use it directly.\n        '
        if size < 0:
            size = INF
        blockid = bisect(self.streamindex, self.pos) - 1
        discard = self.pos - self.streamindex[blockid]
        while size > 0:
            try:
                block_data = self.read_block_data(blockid)
            except EOFError:
                return
            blockid += 1
            if discard != 0:
                if discard >= len(block_data):
                    discard -= len(block_data)
                    continue
                block_data = block_data[discard:]
                discard = 0
            if len(block_data) > size:
                block_data = block_data[:size]
            size -= len(block_data)
            self.pos += len(block_data)
            yield block_data

    def read(self, size: int=-1) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        return b''.join(self.read_blocks(size))

    def get_size(self) -> int:
        if False:
            print('Hello World!')
        del self
        return -1

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.closed = True
        del self.fileobj
        del self.blockoffsets
        del self.streamindex