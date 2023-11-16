from __future__ import print_function
import array
import uuid
import re
import os
import logging
import core.pe as pe

def GetLogger():
    if False:
        for i in range(10):
            print('nop')
    return logging.getLogger('EFI Binary File')

class EFIBinaryError(Exception):

    def __init__(self, message):
        if False:
            for i in range(10):
                print('nop')
        Exception.__init__(self)
        self._message = message

    def GetMessage(self):
        if False:
            for i in range(10):
                print('nop')
        return self._message

class EfiFd(object):
    EFI_FV_HEADER_SIZE = 72

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._fvs = []

    def Load(self, fd, size):
        if False:
            for i in range(10):
                print('nop')
        index = fd.tell()
        while index + self.EFI_FV_HEADER_SIZE < size:
            fv = EfiFv(self)
            fv.Load(fd)
            self._fvs.append(fv)
            index += fv.GetHeader().GetFvLength()
            index = align(index, 8)
            fd.seek(index)

    def GetFvs(self):
        if False:
            while True:
                i = 10
        return self._fvs

class EfiFv(object):
    FILE_SYSTEM_GUID = uuid.UUID('{8c8ce578-8a3d-4f1c-9935-896185c32dd3}')

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        self._size = 0
        self._filename = None
        self._fvheader = None
        self._blockentries = []
        self._ffs = []
        self._parent = parent
        self._offset = 0
        self._raw = array.array('B')

    def Load(self, fd):
        if False:
            print('Hello World!')
        self._offset = fd.tell()
        self._filename = fd.name
        self._fvheader = EfiFirmwareVolumeHeader.Read(fd)
        self._size = self._fvheader.GetFvLength()
        if self._fvheader.GetFileSystemGuid() != self.FILE_SYSTEM_GUID:
            fd.seek(self._offset)
            self._raw.fromfile(fd, self.GetHeader().GetFvLength())
            return
        blockentry = BlockMapEntry.Read(fd)
        self._blockentries.append(blockentry)
        while blockentry.GetNumberBlocks() != 0 and blockentry.GetLength() != 0:
            self._blockentries.append(blockentry)
            blockentry = BlockMapEntry.Read(fd)
        if self._fvheader.GetSize() + len(self._blockentries) * 8 != self._fvheader.GetHeaderLength():
            raise EFIBinaryError('Volume Header length not consistent with block map!')
        index = align(fd.tell(), 8)
        count = 0
        while index + EfiFfs.FFS_HEADER_SIZE < self._size:
            ffs = EfiFfs.Read(fd, self)
            if not isValidGuid(ffs.GetNameGuid()):
                break
            self._ffs.append(ffs)
            count += 1
            index = align(fd.tell(), 8)
        fd.seek(self._offset)
        self._raw.fromfile(fd, self.GetHeader().GetFvLength())

    def GetFfs(self):
        if False:
            while True:
                i = 10
        return self._ffs

    def GetHeader(self):
        if False:
            return 10
        return self._fvheader

    def GetBlockEntries(self):
        if False:
            for i in range(10):
                print('nop')
        return self._blockentries

    def GetHeaderRawData(self):
        if False:
            i = 10
            return i + 15
        ret = []
        ret += self._fvheader.GetRawData()
        for block in self._blockentries:
            ret += block.GetRawData()
        return ret

    def GetOffset(self):
        if False:
            return 10
        return 0

    def GetRawData(self):
        if False:
            for i in range(10):
                print('nop')
        return self._raw.tolist()

class BinaryItem(object):

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        self._size = 0
        self._arr = array.array('B')
        self._parent = parent

    @classmethod
    def Read(cls, fd, parent=None):
        if False:
            i = 10
            return i + 15
        item = cls(parent)
        item.fromfile(fd)
        return item

    def Load(self, fd):
        if False:
            i = 10
            return i + 15
        self.fromfile(fd)

    def GetSize(self):
        if False:
            for i in range(10):
                print('nop')
        'should be implemented by inherited class'

    def fromfile(self, fd):
        if False:
            return 10
        self._arr.fromfile(fd, self.GetSize())

    def GetParent(self):
        if False:
            while True:
                i = 10
        return self._parent

class EfiFirmwareVolumeHeader(BinaryItem):

    def GetSize(self):
        if False:
            for i in range(10):
                print('nop')
        return 56

    def GetSigunature(self):
        if False:
            for i in range(10):
                print('nop')
        list = self._arr.tolist()
        sig = ''
        for x in list[40:44]:
            sig += chr(x)
        return sig

    def GetAttribute(self):
        if False:
            return 10
        return list2int(self._arr.tolist()[44:48])

    def GetErasePolarity(self):
        if False:
            i = 10
            return i + 15
        list = self.GetAttrStrings()
        if 'EFI_FVB2_ERASE_POLARITY' in list:
            return True
        return False

    def GetAttrStrings(self):
        if False:
            for i in range(10):
                print('nop')
        list = []
        value = self.GetAttribute()
        if value & 1 != 0:
            list.append('EFI_FVB2_READ_DISABLED_CAP')
        if value & 2 != 0:
            list.append('EFI_FVB2_READ_ENABLED_CAP')
        if value & 4 != 0:
            list.append('EFI_FVB2_READ_STATUS')
        if value & 8 != 0:
            list.append('EFI_FVB2_WRITE_DISABLED_CAP')
        if value & 16 != 0:
            list.append('EFI_FVB2_WRITE_ENABLED_CAP')
        if value & 32 != 0:
            list.append('EFI_FVB2_WRITE_STATUS')
        if value & 64 != 0:
            list.append('EFI_FVB2_LOCK_CAP')
        if value & 128 != 0:
            list.append('EFI_FVB2_LOCK_STATUS')
        if value & 512 != 0:
            list.append('EFI_FVB2_STICKY_WRITE')
        if value & 1024 != 0:
            list.append('EFI_FVB2_MEMORY_MAPPED')
        if value & 2048 != 0:
            list.append('EFI_FVB2_ERASE_POLARITY')
        if value & 4096 != 0:
            list.append('EFI_FVB2_READ_LOCK_CAP')
        if value & 8192 != 0:
            list.append('EFI_FVB2_READ_LOCK_STATUS')
        if value & 16384 != 0:
            list.append('EFI_FVB2_WRITE_LOCK_CAP')
        if value & 32768 != 0:
            list.append('EFI_FVB2_WRITE_LOCK_STATUS')
        if value == 0:
            list.append('EFI_FVB2_ALIGNMENT_1')
        if value & 2031616 == 65536:
            list.append('EFI_FVB2_ALIGNMENT_2')
        if value & 2031616 == 131072:
            list.append('EFI_FVB2_ALIGNMENT_4')
        if value & 2031616 == 196608:
            list.append('EFI_FVB2_ALIGNMENT_8')
        if value & 2031616 == 262144:
            list.append('EFI_FVB2_ALIGNMENT_16')
        if value & 2031616 == 327680:
            list.append('EFI_FVB2_ALIGNMENT_32')
        if value & 2031616 == 393216:
            list.append('EFI_FVB2_ALIGNMENT_64')
        if value & 2031616 == 458752:
            list.append('EFI_FVB2_ALIGNMENT_128')
        if value & 2031616 == 524288:
            list.append('EFI_FVB2_ALIGNMENT_256')
        if value & 2031616 == 589824:
            list.append('EFI_FVB2_ALIGNMENT_512')
        if value & 2031616 == 655360:
            list.append('EFI_FVB2_ALIGNMENT_1K')
        if value & 2031616 == 720896:
            list.append('EFI_FVB2_ALIGNMENT_2K')
        if value & 2031616 == 786432:
            list.append('EFI_FVB2_ALIGNMENT_4K')
        if value & 2031616 == 851968:
            list.append('EFI_FVB2_ALIGNMENT_8K')
        if value & 2031616 == 917504:
            list.append('EFI_FVB2_ALIGNMENT_16K')
        if value & 2031616 == 983040:
            list.append('EFI_FVB2_ALIGNMENT_32K')
        if value & 2031616 == 1048576:
            list.append('EFI_FVB2_ALIGNMENT_64K')
        if value & 2031616 == 1114112:
            list.append('EFI_FVB2_ALIGNMENT_128K')
        if value & 2031616 == 1179648:
            list.append('EFI_FVB2_ALIGNMENT_256K')
        if value & 2031616 == 1245184:
            list.append('EFI_FVB2_ALIGNMENT_512K')
        return list

    def GetHeaderLength(self):
        if False:
            for i in range(10):
                print('nop')
        return list2int(self._arr.tolist()[48:50])

    def Dump(self):
        if False:
            while True:
                i = 10
        print('Signature: %s' % self.GetSigunature())
        print('Attribute: 0x%X' % self.GetAttribute())
        print('Header Length: 0x%X' % self.GetHeaderLength())
        print('File system Guid: ', self.GetFileSystemGuid())
        print('Revision: 0x%X' % self.GetRevision())
        print('FvLength: 0x%X' % self.GetFvLength())

    def GetFileSystemGuid(self):
        if False:
            for i in range(10):
                print('nop')
        list = self._arr.tolist()
        return list2guid(list[16:32])

    def GetRevision(self):
        if False:
            return 10
        list = self._arr.tolist()
        return int(list[55])

    def GetFvLength(self):
        if False:
            return 10
        list = self._arr.tolist()
        return list2int(list[32:40])

    def GetRawData(self):
        if False:
            while True:
                i = 10
        return self._arr.tolist()

class BlockMapEntry(BinaryItem):

    def GetSize(self):
        if False:
            i = 10
            return i + 15
        return 8

    def GetNumberBlocks(self):
        if False:
            i = 10
            return i + 15
        list = self._arr.tolist()
        return list2int(list[0:4])

    def GetLength(self):
        if False:
            print('Hello World!')
        list = self._arr.tolist()
        return list2int(list[4:8])

    def GetRawData(self):
        if False:
            i = 10
            return i + 15
        return self._arr.tolist()

    def __str__(self):
        if False:
            while True:
                i = 10
        return '[BlockEntry] Number = 0x%X, length=0x%X' % (self.GetNumberBlocks(), self.GetLength())

class EfiFfs(object):
    FFS_HEADER_SIZE = 24

    def __init__(self, parent=None):
        if False:
            return 10
        self._header = None
        self._parent = parent
        self._offset = 0
        self._sections = []

    def Load(self, fd):
        if False:
            print('Hello World!')
        self._offset = align(fd.tell(), 8)
        self._header = EfiFfsHeader.Read(fd, self)
        if not isValidGuid(self.GetNameGuid()):
            return
        index = self._offset
        fileend = self._offset + self.GetSize()
        while index + EfiSection.EFI_SECTION_HEADER_SIZE < fileend:
            section = EfiSection(self)
            section.Load(fd)
            if section.GetSize() == 0 and section.GetHeader().GetType() == 0:
                break
            self._sections.append(section)
            index = fd.tell()
        index = self._offset + self._header.GetFfsSize()
        index = align(index, 8)
        fd.seek(index)

    def GetOffset(self):
        if False:
            return 10
        return self._offset

    def GetSize(self):
        if False:
            print('Hello World!')
        return self._header.GetFfsSize()

    @classmethod
    def Read(cls, fd, parent=None):
        if False:
            return 10
        item = cls(parent)
        item.Load(fd)
        return item

    def GetNameGuid(self):
        if False:
            for i in range(10):
                print('nop')
        return self._header.GetNameGuid()

    def DumpContent(self):
        if False:
            return 10
        list = self._content.tolist()
        line = []
        count = 0
        for item in list:
            if count < 32:
                line.append('0x%X' % int(item))
                count += 1
            else:
                print(' '.join(line))
                count = 0
                line = []
                line.append('0x%X' % int(item))
                count += 1

    def GetHeader(self):
        if False:
            return 10
        return self._header

    def GetParent(self):
        if False:
            for i in range(10):
                print('nop')
        return self._parent

    def GetSections(self):
        if False:
            i = 10
            return i + 15
        return self._sections

class EfiFfsHeader(BinaryItem):
    ffs_state_map = {1: 'EFI_FILE_HEADER_CONSTRUCTION', 2: 'EFI_FILE_HEADER_VALID', 4: 'EFI_FILE_DATA_VALID', 8: 'EFI_FILE_MARKED_FOR_UPDATE', 16: 'EFI_FILE_DELETED', 32: 'EFI_FILE_HEADER_INVALID'}

    def GetSize(self):
        if False:
            return 10
        return 24

    def GetNameGuid(self):
        if False:
            for i in range(10):
                print('nop')
        list = self._arr.tolist()
        return list2guid(list[0:16])

    def GetType(self):
        if False:
            i = 10
            return i + 15
        list = self._arr.tolist()
        return int(list[18])

    def GetTypeString(self):
        if False:
            for i in range(10):
                print('nop')
        value = self.GetType()
        if value == 1:
            return 'EFI_FV_FILETYPE_RAW'
        if value == 2:
            return 'EFI_FV_FILETYPE_FREEFORM'
        if value == 3:
            return 'EFI_FV_FILETYPE_SECURITY_CORE'
        if value == 4:
            return 'EFI_FV_FILETYPE_PEI_CORE'
        if value == 5:
            return 'EFI_FV_FILETYPE_DXE_CORE'
        if value == 6:
            return 'EFI_FV_FILETYPE_PEIM'
        if value == 7:
            return 'EFI_FV_FILETYPE_DRIVER'
        if value == 8:
            return 'EFI_FV_FILETYPE_COMBINED_PEIM_DRIVER'
        if value == 9:
            return 'EFI_FV_FILETYPE_APPLICATION'
        if value == 11:
            return 'EFI_FV_FILETYPE_FIRMWARE_VOLUME_IMAGE'
        if value == 192:
            return 'EFI_FV_FILETYPE_OEM_MIN'
        if value == 223:
            return 'EFI_FV_FILETYPE_OEM_MAX'
        if value == 224:
            return 'EFI_FV_FILETYPE_DEBUG_MIN'
        if value == 239:
            return 'EFI_FV_FILETYPE_DEBUG_MAX'
        if value == 240:
            return 'EFI_FV_FILETYPE_FFS_PAD'
        if value == 255:
            return 'EFI_FV_FILETYPE_FFS_MAX'
        return 'Unknown FFS Type'

    def GetAttributes(self):
        if False:
            print('Hello World!')
        list = self._arr.tolist()
        return int(list[19])

    def GetFfsSize(self):
        if False:
            while True:
                i = 10
        list = self._arr.tolist()
        return list2int(list[20:23])

    def GetState(self):
        if False:
            for i in range(10):
                print('nop')
        list = self._arr.tolist()
        state = int(list[23])
        polarity = self.GetParent().GetParent().GetHeader().GetErasePolarity()
        if polarity:
            state = ~state & 255
        HighestBit = 128
        while HighestBit != 0 and HighestBit & state == 0:
            HighestBit = HighestBit >> 1
        return HighestBit

    def GetStateString(self):
        if False:
            while True:
                i = 10
        state = self.GetState()
        if state in self.ffs_state_map.keys():
            return self.ffs_state_map[state]
        return 'Unknown Ffs State'

    def Dump(self):
        if False:
            print('Hello World!')
        print('FFS name: ', self.GetNameGuid())
        print('FFS type: ', self.GetType())
        print('FFS attr: 0x%X' % self.GetAttributes())
        print('FFS size: 0x%X' % self.GetFfsSize())
        print('FFS state: 0x%X' % self.GetState())

    def GetRawData(self):
        if False:
            while True:
                i = 10
        return self._arr.tolist()

class EfiSection(object):
    EFI_SECTION_HEADER_SIZE = 4

    def __init__(self, parent=None):
        if False:
            return 10
        self._size = 0
        self._parent = parent
        self._offset = 0
        self._contents = array.array('B')

    def Load(self, fd):
        if False:
            for i in range(10):
                print('nop')
        self._offset = align(fd.tell(), 4)
        self._header = EfiSectionHeader.Read(fd, self)
        if self._header.GetTypeString() == 'EFI_SECTION_PE32':
            pefile = pe.PEFile(self)
            pefile.Load(fd, self.GetContentSize())
        fd.seek(self._offset)
        self._contents.fromfile(fd, self.GetContentSize())
        index = self._offset + self.GetSize()
        index = align(index, 4)
        fd.seek(index)

    def GetContentSize(self):
        if False:
            while True:
                i = 10
        return self.GetSize() - self.EFI_SECTION_HEADER_SIZE

    def GetContent(self):
        if False:
            while True:
                i = 10
        return self._contents.tolist()

    def GetSize(self):
        if False:
            for i in range(10):
                print('nop')
        return self._header.GetSectionSize()

    def GetHeader(self):
        if False:
            return 10
        return self._header

    def GetSectionOffset(self):
        if False:
            for i in range(10):
                print('nop')
        return self._offset + self.EFI_SECTION_HEADER_SIZE

class EfiSectionHeader(BinaryItem):
    section_type_map = {1: 'EFI_SECTION_COMPRESSION', 2: 'EFI_SECTION_GUID_DEFINED', 16: 'EFI_SECTION_PE32', 17: 'EFI_SECTION_PIC', 18: 'EFI_SECTION_TE', 19: 'EFI_SECTION_DXE_DEPEX', 20: 'EFI_SECTION_VERSION', 21: 'EFI_SECTION_USER_INTERFACE', 22: 'EFI_SECTION_COMPATIBILITY16', 23: 'EFI_SECTION_FIRMWARE_VOLUME_IMAGE', 24: 'EFI_SECTION_FREEFORM_SUBTYPE_GUID', 25: 'EFI_SECTION_RAW', 27: 'EFI_SECTION_PEI_DEPEX'}

    def GetSize(self):
        if False:
            i = 10
            return i + 15
        return 4

    def GetSectionSize(self):
        if False:
            return 10
        list = self._arr.tolist()
        return list2int(list[0:3])

    def GetType(self):
        if False:
            return 10
        list = self._arr.tolist()
        return int(list[3])

    def GetTypeString(self):
        if False:
            while True:
                i = 10
        type = self.GetType()
        if type not in self.section_type_map.keys():
            return 'Unknown Section Type'
        return self.section_type_map[type]

    def Dump(self):
        if False:
            i = 10
            return i + 15
        print('size = 0x%X' % self.GetSectionSize())
        print('type = 0x%X' % self.GetType())
rMapEntry = re.compile('^(\\w+)[ \\(\\w\\)]* \\(BaseAddress=([0-9a-fA-F]+), EntryPoint=([0-9a-fA-F]+), GUID=([0-9a-fA-F\\-]+)')

class EfiFvMapFile(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._mapentries = {}

    def Load(self, path):
        if False:
            return 10
        if not os.path.exists(path):
            return False
        try:
            file = open(path, 'r')
            lines = file.readlines()
            file.close()
        except:
            return False
        for line in lines:
            if line[0] != ' ':
                ret = rMapEntry.match(line)
                if ret is not None:
                    name = ret.groups()[0]
                    baseaddr = int(ret.groups()[1], 16)
                    entry = int(ret.groups()[2], 16)
                    guidstr = '{' + ret.groups()[3] + '}'
                    guid = uuid.UUID(guidstr)
                    self._mapentries[guid] = EfiFvMapFileEntry(name, baseaddr, entry, guid)
        return True

    def GetEntry(self, guid):
        if False:
            i = 10
            return i + 15
        if guid in self._mapentries.keys():
            return self._mapentries[guid]
        return None

class EfiFvMapFileEntry(object):

    def __init__(self, name, baseaddr, entry, guid):
        if False:
            while True:
                i = 10
        self._name = name
        self._baseaddr = baseaddr
        self._entry = entry
        self._guid = guid

    def GetName(self):
        if False:
            print('Hello World!')
        return self._name

    def GetBaseAddress(self):
        if False:
            return 10
        return self._baseaddr

    def GetEntryPoint(self):
        if False:
            for i in range(10):
                print('nop')
        return self._entry

def list2guid(list):
    if False:
        while True:
            i = 10
    val1 = list2int(list[0:4])
    val2 = list2int(list[4:6])
    val3 = list2int(list[6:8])
    val4 = 0
    for item in list[8:16]:
        val4 = val4 << 8 | int(item)
    val = val1 << 12 * 8 | val2 << 10 * 8 | val3 << 8 * 8 | val4
    guid = uuid.UUID(int=val)
    return guid

def list2int(list):
    if False:
        return 10
    val = 0
    for index in range(len(list) - 1, -1, -1):
        val = val << 8 | int(list[index])
    return val

def align(value, alignment):
    if False:
        i = 10
        return i + 15
    return value + (alignment - value & alignment - 1)
gInvalidGuid = uuid.UUID(int=340282366920938463463374607431768211455)

def isValidGuid(guid):
    if False:
        print('Hello World!')
    if guid == gInvalidGuid:
        return False
    return True