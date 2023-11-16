import argparse
from ctypes import *
import struct

class ElfSectionHeader64:

    def __init__(self, sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize):
        if False:
            i = 10
            return i + 15
        self.sh_name = sh_name
        self.sh_type = sh_type
        self.sh_flags = sh_flags
        self.sh_addr = sh_addr
        self.sh_offset = sh_offset
        self.sh_size = sh_size
        self.sh_link = sh_link
        self.sh_info = sh_info
        self.sh_addralign = sh_addralign
        self.sh_entsize = sh_entsize

    def pack(self):
        if False:
            while True:
                i = 10
        return struct.pack('<IIQQQQIIQQ', self.sh_name, self.sh_type, self.sh_flags, self.sh_addr, self.sh_offset, self.sh_size, self.sh_link, self.sh_info, self.sh_addralign, self.sh_entsize)

    @classmethod
    def unpack(cls, data):
        if False:
            i = 10
            return i + 15
        unpacked_data = struct.unpack('<IIQQQQIIQQ', data)
        return cls(*unpacked_data)

class ElfHeader64:

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.e_ident = struct.unpack('16s', data[:16])[0]
        self.e_type = struct.unpack('H', data[16:18])[0]
        self.e_machine = struct.unpack('H', data[18:20])[0]
        self.e_version = struct.unpack('I', data[20:24])[0]
        self.e_entry = struct.unpack('Q', data[24:32])[0]
        self.e_phoff = struct.unpack('Q', data[32:40])[0]
        self.e_shoff = struct.unpack('Q', data[40:48])[0]
        self.e_flags = struct.unpack('I', data[48:52])[0]
        self.e_ehsize = struct.unpack('H', data[52:54])[0]
        self.e_phentsize = struct.unpack('H', data[54:56])[0]
        self.e_phnum = struct.unpack('H', data[56:58])[0]
        self.e_shentsize = struct.unpack('H', data[58:60])[0]
        self.e_shnum = struct.unpack('H', data[60:62])[0]
        self.e_shstrndx = struct.unpack('H', data[62:64])[0]

    def pack(self):
        if False:
            while True:
                i = 10
        data = b''
        data += struct.pack('16s', self.e_ident)
        data += struct.pack('H', self.e_type)
        data += struct.pack('H', self.e_machine)
        data += struct.pack('I', self.e_version)
        data += struct.pack('Q', self.e_entry)
        data += struct.pack('Q', self.e_phoff)
        data += struct.pack('Q', self.e_shoff)
        data += struct.pack('I', self.e_flags)
        data += struct.pack('H', self.e_ehsize)
        data += struct.pack('H', self.e_phentsize)
        data += struct.pack('H', self.e_phnum)
        data += struct.pack('H', self.e_shentsize)
        data += struct.pack('H', self.e_shnum)
        data += struct.pack('H', self.e_shstrndx)
        return data

class Elf64_Phdr:

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self.p_type = struct.unpack('<L', data[0:4])[0]
        self.p_flags = struct.unpack('<L', data[4:8])[0]
        self.p_offset = struct.unpack('<Q', data[8:16])[0]
        self.p_vaddr = struct.unpack('<Q', data[16:24])[0]
        self.p_paddr = struct.unpack('<Q', data[24:32])[0]
        self.p_filesz = struct.unpack('<Q', data[32:40])[0]
        self.p_memsz = struct.unpack('<Q', data[40:48])[0]
        self.p_align = struct.unpack('<Q', data[48:56])[0]

    def pack(self):
        if False:
            for i in range(10):
                print('nop')
        data = b''
        data += struct.pack('<L', self.p_type)
        data += struct.pack('<L', self.p_flags)
        data += struct.pack('<Q', self.p_offset)
        data += struct.pack('<Q', self.p_vaddr)
        data += struct.pack('<Q', self.p_paddr)
        data += struct.pack('<Q', self.p_filesz)
        data += struct.pack('<Q', self.p_memsz)
        data += struct.pack('<Q', self.p_align)
        return data

class ElfSectionHeader32:

    def __init__(self, sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize):
        if False:
            while True:
                i = 10
        self.sh_name = sh_name
        self.sh_type = sh_type
        self.sh_flags = sh_flags
        self.sh_addr = sh_addr
        self.sh_offset = sh_offset
        self.sh_size = sh_size
        self.sh_link = sh_link
        self.sh_info = sh_info
        self.sh_addralign = sh_addralign
        self.sh_entsize = sh_entsize

    def pack(self):
        if False:
            while True:
                i = 10
        return struct.pack('<IIIIIIIIII', self.sh_name, self.sh_type, self.sh_flags, self.sh_addr, self.sh_offset, self.sh_size, self.sh_link, self.sh_info, self.sh_addralign, self.sh_entsize)

    @classmethod
    def unpack(cls, data):
        if False:
            i = 10
            return i + 15
        unpacked_data = struct.unpack('<IIIIIIIIII', data)
        return cls(*unpacked_data)

class ElfHeader32:

    def __init__(self, data):
        if False:
            return 10
        self.e_ident = struct.unpack('16s', data[:16])[0]
        self.e_type = struct.unpack('H', data[16:18])[0]
        self.e_machine = struct.unpack('H', data[18:20])[0]
        self.e_version = struct.unpack('I', data[20:24])[0]
        self.e_entry = struct.unpack('I', data[24:28])[0]
        self.e_phoff = struct.unpack('I', data[28:32])[0]
        self.e_shoff = struct.unpack('I', data[32:36])[0]
        self.e_flags = struct.unpack('I', data[36:40])[0]
        self.e_ehsize = struct.unpack('H', data[40:42])[0]
        self.e_phentsize = struct.unpack('H', data[42:44])[0]
        self.e_phnum = struct.unpack('H', data[44:46])[0]
        self.e_shentsize = struct.unpack('H', data[46:48])[0]
        self.e_shnum = struct.unpack('H', data[48:50])[0]
        self.e_shstrndx = struct.unpack('H', data[50:52])[0]

    def pack(self):
        if False:
            print('Hello World!')
        data = b''
        data += struct.pack('16s', self.e_ident)
        data += struct.pack('H', self.e_type)
        data += struct.pack('H', self.e_machine)
        data += struct.pack('I', self.e_version)
        data += struct.pack('I', self.e_entry)
        data += struct.pack('I', self.e_phoff)
        data += struct.pack('I', self.e_shoff)
        data += struct.pack('I', self.e_flags)
        data += struct.pack('H', self.e_ehsize)
        data += struct.pack('H', self.e_phentsize)
        data += struct.pack('H', self.e_phnum)
        data += struct.pack('H', self.e_shentsize)
        data += struct.pack('H', self.e_shnum)
        data += struct.pack('H', self.e_shstrndx)
        return data

class Elf32_Phdr:

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.p_type = struct.unpack('<L', data[0:4])[0]
        self.p_offset = struct.unpack('<L', data[4:8])[0]
        self.p_vaddr = struct.unpack('<L', data[8:12])[0]
        self.p_paddr = struct.unpack('<L', data[12:16])[0]
        self.p_filesz = struct.unpack('<L', data[16:20])[0]
        self.p_memsz = struct.unpack('<L', data[20:24])[0]
        self.p_flags = struct.unpack('<L', data[24:28])[0]
        self.p_align = struct.unpack('<L', data[28:32])[0]

    def pack(self):
        if False:
            print('Hello World!')
        data = b''
        data += struct.pack('<L', self.p_type)
        data += struct.pack('<L', self.p_offset)
        data += struct.pack('<L', self.p_vaddr)
        data += struct.pack('<L', self.p_paddr)
        data += struct.pack('<L', self.p_filesz)
        data += struct.pack('<L', self.p_memsz)
        data += struct.pack('<L', self.p_flags)
        data += struct.pack('<L', self.p_align)
        return data

def SectionAlignment(NewUPLEntry, AlignmentIndex):
    if False:
        i = 10
        return i + 15
    if isinstance(AlignmentIndex, str):
        int_num = int(AlignmentIndex, 16)
        int_num = 10 * (int_num // 16) + int_num % 16
    else:
        int_num = AlignmentIndex
    if int_num != 0 or int_num != 1:
        if len(NewUPLEntry) % int_num != 0:
            AlignNumber = int_num - len(NewUPLEntry) % int_num
            if AlignNumber != 0:
                for x in range(AlignNumber):
                    NewUPLEntry = NewUPLEntry + bytearray(b'\x00')
    return NewUPLEntry

def SectionEntryFill(SectionEntry, Alignment, Value, Offset):
    if False:
        while True:
            i = 10
    n = 0
    if len(Value) < Alignment:
        Value = Value.zfill(Alignment)
    for x in range(0, Alignment // 2):
        Index = '0x' + Value[n] + Value[n + 1]
        SectionEntry[Offset - x] = int(Index, 16)
        n += 2
    return SectionEntry

def ElfHeaderParser(UPLEntry):
    if False:
        while True:
            i = 10
    EI_CLASS = UPLEntry[4]
    if EI_CLASS == 2:
        ElfHeaderData = UPLEntry[:64]
    else:
        ElfHeaderData = UPLEntry[:53]
    if EI_CLASS == 2:
        elf_header = ElfHeader64(ElfHeaderData)
        ElfHeaderOffset = elf_header.e_shoff
        SectionHeaderEntryNumber = elf_header.e_shnum
        StringIndexNumber = elf_header.e_shstrndx
        SectionHeaderEntrySize = elf_header.e_shentsize
        StringIndexEntryOffset = ElfHeaderOffset + StringIndexNumber * SectionHeaderEntrySize
        unpacked_header = ElfSectionHeader64.unpack(UPLEntry[StringIndexEntryOffset:StringIndexEntryOffset + SectionHeaderEntrySize])
        StringIndexSize = unpacked_header.sh_size
        StringIndexOffset = unpacked_header.sh_offset
    else:
        elf_header = ElfHeader32(ElfHeaderData)
        ElfHeaderOffset = elf_header.e_shoff
        SectionHeaderEntryNumber = elf_header.e_shnum
        StringIndexNumber = elf_header.e_shstrndx
        SectionHeaderEntrySize = elf_header.e_shentsize
        StringIndexEntryOffset = ElfHeaderOffset + StringIndexNumber * SectionHeaderEntrySize
        unpacked_header = ElfSectionHeader32.unpack(UPLEntry[StringIndexEntryOffset:StringIndexEntryOffset + SectionHeaderEntrySize])
        StringIndexSize = unpacked_header.sh_size
        StringIndexOffset = unpacked_header.sh_offset
    return (ElfHeaderOffset, SectionHeaderEntryNumber, StringIndexNumber, StringIndexEntryOffset, StringIndexSize, SectionHeaderEntrySize, StringIndexOffset, EI_CLASS)

def FindSection(UPLEntry, SectionName):
    if False:
        return 10
    (ElfHeaderOffset, SectionHeaderEntryNumber, StringIndexNumber, _, StringIndexSize, SectionHeaderEntrySize, StringIndexOffset, EI_CLASS) = ElfHeaderParser(UPLEntry)
    StringIndex = UPLEntry[StringIndexOffset:StringIndexOffset + StringIndexSize]
    StringIndex = StringIndex.decode('utf-8', errors='ignore')
    SectionNameOffset = StringIndex.find(SectionName)
    return (SectionNameOffset, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, StringIndexOffset, StringIndexNumber, EI_CLASS)

def AddNewSectionEntry64(LastUPLEntrylen, StringIndexValue, SectionSize, Alignment):
    if False:
        return 10
    NewSectionEntry = ElfSectionHeader64(StringIndexValue, 1, 0, 0, LastUPLEntrylen, SectionSize, 0, 0, Alignment, 0)
    sh_bytes = NewSectionEntry.pack()
    return sh_bytes

def AddNewSectionEntry32(LastUPLEntrylen, StringIndexValue, SectionSize, Alignment):
    if False:
        print('Hello World!')
    NewSectionEntry = ElfSectionHeader32(StringIndexValue, 1, 0, 0, LastUPLEntrylen, SectionSize, 0, 0, Alignment, 0)
    sh_bytes = NewSectionEntry.pack()
    return sh_bytes

def AddSectionHeader64(SHentry, NewUPLEntrylen, SectionHeaderEntrySize, Index, RemoveNameOffset, SectionName, StringIndexNumber):
    if False:
        return 10
    SHentry = bytearray(SHentry)
    unpacked_header = ElfSectionHeader64.unpack(SHentry[Index * SectionHeaderEntrySize:Index * SectionHeaderEntrySize + SectionHeaderEntrySize])
    if Index != 0:
        unpacked_header.sh_offset = NewUPLEntrylen
        if RemoveNameOffset != 0:
            if unpacked_header.sh_name > RemoveNameOffset:
                unpacked_header.sh_name -= len(SectionName)
            if Index == StringIndexNumber:
                unpacked_header.sh_size -= len(SectionName)
        elif Index == StringIndexNumber:
            unpacked_header.sh_size += len(SectionName)
    NewSHentry = ElfSectionHeader64(unpacked_header.sh_name, unpacked_header.sh_type, unpacked_header.sh_flags, unpacked_header.sh_addr, unpacked_header.sh_offset, unpacked_header.sh_size, unpacked_header.sh_link, unpacked_header.sh_info, unpacked_header.sh_addralign, unpacked_header.sh_entsize).pack()
    return NewSHentry

def AddSectionHeader32(SHentry, NewUPLEntrylen, SectionHeaderEntrySize, Index, RemoveNameOffset, SectionName, StringIndexNumber):
    if False:
        while True:
            i = 10
    SHentry = bytearray(SHentry)
    unpacked_header = ElfSectionHeader32.unpack(SHentry[Index * SectionHeaderEntrySize:Index * SectionHeaderEntrySize + SectionHeaderEntrySize])
    if Index != 0:
        NewSHentry = SHentry[Index * SectionHeaderEntrySize:Index * SectionHeaderEntrySize + SectionHeaderEntrySize]
        unpacked_header.sh_offset = NewUPLEntrylen
        if RemoveNameOffset != 0:
            if unpacked_header.sh_name > RemoveNameOffset:
                unpacked_header.sh_name -= len(SectionName)
            if Index == StringIndexNumber:
                unpacked_header.sh_size -= len(SectionName)
        elif Index == StringIndexNumber:
            unpacked_header.sh_size += len(SectionName)
    NewSHentry = ElfSectionHeader32(unpacked_header.sh_name, unpacked_header.sh_type, unpacked_header.sh_flags, unpacked_header.sh_addr, unpacked_header.sh_offset, unpacked_header.sh_size, unpacked_header.sh_link, unpacked_header.sh_info, unpacked_header.sh_addralign, unpacked_header.sh_entsize).pack()
    return NewSHentry

def ModifyPHSegmentOffset64(NewUPLEntry, ElfHeaderOffset, PHSegmentName):
    if False:
        return 10
    elf_header = ElfHeader64(NewUPLEntry[:64])
    SHentry = NewUPLEntry[ElfHeaderOffset:]
    PHentry = NewUPLEntry[64:64 + elf_header.e_phnum * elf_header.e_phentsize]
    PHdrs = []
    SHdrs = []
    for i in range(elf_header.e_shnum):
        SHData = SHentry[i * elf_header.e_shentsize:i * elf_header.e_shentsize + elf_header.e_shentsize]
        unpacked_SectionHeader = ElfSectionHeader64.unpack(SHData)
        SHdrs.append(unpacked_SectionHeader)
    for i in range(elf_header.e_phnum):
        PHData = PHentry[i * elf_header.e_phentsize:i * elf_header.e_phentsize + elf_header.e_phentsize]
        unpacked_ProgramHeader = Elf64_Phdr(PHData)
        PHdrs.append(unpacked_ProgramHeader)
    if PHSegmentName == '.text':
        PHdrs[0].p_offset = SHdrs[1].sh_offset
        PHdrs[0].p_paddr = SHdrs[1].sh_addr
        PHdrs[4].p_offset = SHdrs[1].sh_offset
        PHdrs[4].p_paddr = SHdrs[1].sh_addr
    elif PHSegmentName == '.dynamic':
        PHdrs[1].p_offset = SHdrs[2].sh_offset
        PHdrs[1].p_paddr = SHdrs[2].sh_addr
        PHdrs[3].p_offset = SHdrs[2].sh_offset
        PHdrs[3].p_paddr = SHdrs[2].sh_addr
    elif PHSegmentName == '.data':
        PHdrs[2].p_offset = SHdrs[3].sh_offset
        PHdrs[2].p_paddr = SHdrs[3].sh_addr
    packed_PHData = b''
    for phdr in PHdrs:
        packed_PHData += phdr.pack()
    NewUPLEntry = bytearray(NewUPLEntry)
    NewUPLEntry[64:64 + elf_header.e_phnum * elf_header.e_phentsize] = packed_PHData
    return NewUPLEntry

def ModifyPHSegmentOffset32(NewUPLEntry, ElfHeaderOffset, PHSegmentName):
    if False:
        while True:
            i = 10
    elf_header = ElfHeader32(NewUPLEntry[:52])
    SHentry = NewUPLEntry[ElfHeaderOffset:]
    PHentry = NewUPLEntry[52:52 + elf_header.e_phnum * elf_header.e_phentsize]
    PHdrs = []
    SHdrs = []
    for i in range(elf_header.e_shnum):
        SHData = SHentry[i * elf_header.e_shentsize:i * elf_header.e_shentsize + elf_header.e_shentsize]
        unpacked_SectionHeader = ElfSectionHeader32.unpack(SHData)
        SHdrs.append(unpacked_SectionHeader)
    for i in range(elf_header.e_phnum):
        PHData = PHentry[i * elf_header.e_phentsize:i * elf_header.e_phentsize + elf_header.e_phentsize]
        unpacked_ProgramHeader = Elf32_Phdr(PHData)
        PHdrs.append(unpacked_ProgramHeader)
    if PHSegmentName == '.text':
        PHdrs[0].p_offset = SHdrs[1].sh_offset
        PHdrs[0].p_paddr = SHdrs[1].sh_addr
        PHdrs[0].p_vaddr = SHdrs[1].sh_addr
        PHdrs[2].p_offset = SHdrs[1].sh_offset
        PHdrs[2].p_paddr = SHdrs[1].sh_addr
        PHdrs[0].p_vaddr = SHdrs[1].sh_addr
    elif PHSegmentName == '.data':
        PHdrs[1].p_offset = SHdrs[2].sh_offset
        PHdrs[1].p_paddr = SHdrs[2].sh_addr
        PHdrs[1].p_vaddr = SHdrs[2].sh_addr
    packed_PHData = b''
    for phdr in PHdrs:
        packed_PHData += phdr.pack()
    NewUPLEntry = bytearray(NewUPLEntry)
    NewUPLEntry[52:52 + elf_header.e_phnum * elf_header.e_phentsize] = packed_PHData
    return NewUPLEntry

def RemoveSection64(UniversalPayloadEntry, RemoveSectionName):
    if False:
        i = 10
        return i + 15
    with open(UniversalPayloadEntry, 'rb') as f:
        UPLEntry = f.read()
        (RemoveSectionNameOffset, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, _, StringIndexNumber, _) = FindSection(UPLEntry, RemoveSectionName)
        if RemoveSectionNameOffset == -1:
            raise argparse.ArgumentTypeError('Section: {} not found.'.format(RemoveSectionNameOffset))
        SHentry = UPLEntry[ElfHeaderOffset:]
        elf_header = ElfHeader64(UPLEntry[:64])
        Counter = 0
        RemoveIndex = 0
        RemoveNameOffset = 0
        for Index in range(0, elf_header.e_shnum):
            unpacked_SectionHeader = ElfSectionHeader64.unpack(SHentry[Index * elf_header.e_shentsize:Index * elf_header.e_shentsize + elf_header.e_shentsize])
            if unpacked_SectionHeader.sh_name == RemoveSectionNameOffset:
                RemoveIndex = Counter
                Counter += 1
            else:
                Counter += 1
        ElfHeaderSize = 64
        ElfHandPH = ElfHeaderSize + elf_header.e_phnum * elf_header.e_phentsize
        NewUPLEntry = UPLEntry[:ElfHandPH]
        NewUPLEntry = bytearray(NewUPLEntry)
        NewUPLEntrylen = []
        for Index in range(0, SectionHeaderEntryNumber):
            unpacked_SectionHeader = ElfSectionHeader64.unpack(SHentry[Index * SectionHeaderEntrySize:Index * SectionHeaderEntrySize + SectionHeaderEntrySize])
            NewUPLEntrylen.append(len(NewUPLEntry))
            if Index == 0:
                AlignmentIndex = 8
                if SectionHeaderEntryNumber > 2:
                    unpacked_NextSectionHeader = ElfSectionHeader64.unpack(SHentry[(Index + 1) * SectionHeaderEntrySize:(Index + 1) * SectionHeaderEntrySize + SectionHeaderEntrySize])
                NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_NextSectionHeader.sh_addralign)
            elif Index + 1 == RemoveIndex:
                NewUPLEntry += UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                if Index + 2 < SectionHeaderEntryNumber - 1:
                    unpacked_Next2SectionHeader = ElfSectionHeader64.unpack(SHentry[(Index + 2) * SectionHeaderEntrySize:(Index + 2) * SectionHeaderEntrySize + SectionHeaderEntrySize])
                    NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_Next2SectionHeader.sh_addralign)
                else:
                    AlignmentIndex = 8
                    NewUPLEntry = SectionAlignment(NewUPLEntry, AlignmentIndex)
            elif Index == RemoveIndex:
                RemoveNameOffset = unpacked_SectionHeader.sh_name
            elif Index == StringIndexNumber:
                StringIndex = UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                StringIndex = bytearray(StringIndex)
                RemoveSectionName = bytearray(RemoveSectionName, encoding='utf-8')
                RemoveSectionName = RemoveSectionName + bytes('\x00', encoding='utf-8')
                StringIndex = StringIndex.replace(RemoveSectionName, b'')
                NewUPLEntry += StringIndex
            else:
                NewUPLEntry += UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                if Index < SectionHeaderEntryNumber - 1:
                    NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_NextSectionHeader.sh_addralign)
                else:
                    AlignmentIndex = 8
                    NewUPLEntry = SectionAlignment(NewUPLEntry, AlignmentIndex)
        SectionHeaderOffset = len(NewUPLEntry)
        for Number in range(0, SectionHeaderEntryNumber):
            if Number != RemoveIndex:
                NewSHentry = AddSectionHeader64(SHentry, NewUPLEntrylen[Number], SectionHeaderEntrySize, Number, RemoveNameOffset, RemoveSectionName, StringIndexNumber)
                NewUPLEntry += NewSHentry
        elf_header.e_shoff = SectionHeaderOffset
        elf_header.e_shnum -= 1
        NewUPLEntry = elf_header.pack() + NewUPLEntry[64:]
        with open(UniversalPayloadEntry, 'wb') as f:
            f.write(NewUPLEntry)

def RemoveSection32(UniversalPayloadEntry, RemoveSectionName):
    if False:
        print('Hello World!')
    with open(UniversalPayloadEntry, 'rb') as f:
        UPLEntry = f.read()
        (RemoveSectionNameOffset, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, _, StringIndexNumber, EI_CLASS) = FindSection(UPLEntry, RemoveSectionName)
        if RemoveSectionNameOffset == -1:
            raise argparse.ArgumentTypeError('Section: {} not found.'.format(RemoveSectionNameOffset))
        SHentry = UPLEntry[ElfHeaderOffset:]
        elf_header = ElfHeader32(UPLEntry[:52])
        Counter = 0
        RemoveIndex = 0
        RemoveNameOffset = 0
        for Index in range(0, elf_header.e_shnum):
            unpacked_SectionHeader = ElfSectionHeader32.unpack(SHentry[Index * elf_header.e_shentsize:Index * elf_header.e_shentsize + elf_header.e_shentsize])
            if unpacked_SectionHeader.sh_name == RemoveSectionNameOffset:
                RemoveIndex = Counter
                Counter += 1
            else:
                Counter += 1
        ElfHeaderSize = 52
        ElfHandPH = ElfHeaderSize + elf_header.e_phnum * elf_header.e_phentsize
        NewUPLEntry = UPLEntry[:ElfHandPH]
        NewUPLEntry = bytearray(NewUPLEntry)
        NewUPLEntrylen = []
        for Index in range(0, SectionHeaderEntryNumber):
            unpacked_SectionHeader = ElfSectionHeader32.unpack(SHentry[Index * SectionHeaderEntrySize:Index * SectionHeaderEntrySize + SectionHeaderEntrySize])
            NewUPLEntrylen.append(len(NewUPLEntry))
            if Index == 0:
                AlignmentIndex = 8
                if SectionHeaderEntryNumber > 2:
                    unpacked_NextSectionHeader = ElfSectionHeader32.unpack(SHentry[(Index + 1) * SectionHeaderEntrySize:(Index + 1) * SectionHeaderEntrySize + SectionHeaderEntrySize])
                NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_NextSectionHeader.sh_addralign)
            elif Index + 1 == RemoveIndex:
                NewUPLEntry += UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                if Index + 2 < SectionHeaderEntryNumber - 1:
                    unpacked_Next2SectionHeader = ElfSectionHeader32.unpack(SHentry[(Index + 2) * SectionHeaderEntrySize:(Index + 2) * SectionHeaderEntrySize + SectionHeaderEntrySize])
                    NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_Next2SectionHeader.sh_addralign)
                else:
                    AlignmentIndex = 8
                    NewUPLEntry = SectionAlignment(NewUPLEntry, AlignmentIndex)
            elif Index == RemoveIndex:
                RemoveNameOffset = unpacked_SectionHeader.sh_name
            elif Index == StringIndexNumber:
                StringIndex = UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                StringIndex = bytearray(StringIndex)
                RemoveSectionName = bytearray(RemoveSectionName, encoding='utf-8')
                RemoveSectionName = RemoveSectionName + bytes('\x00', encoding='utf-8')
                StringIndex = StringIndex.replace(RemoveSectionName, b'')
                NewUPLEntry += StringIndex
            else:
                NewUPLEntry += UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                if Index < SectionHeaderEntryNumber - 1:
                    NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_NextSectionHeader.sh_addralign)
                else:
                    AlignmentIndex = 8
                    NewUPLEntry = SectionAlignment(NewUPLEntry, AlignmentIndex)
        SectionHeaderOffset = len(NewUPLEntry)
        for Number in range(0, SectionHeaderEntryNumber):
            if Number != RemoveIndex:
                NewSHentry = AddSectionHeader32(SHentry, NewUPLEntrylen[Number], SectionHeaderEntrySize, Number, RemoveNameOffset, RemoveSectionName, StringIndexNumber)
                NewUPLEntry += NewSHentry
        elf_header.e_shoff = SectionHeaderOffset
        elf_header.e_shnum -= 1
        NewUPLEntry = elf_header.pack() + NewUPLEntry[52:]
        with open(UniversalPayloadEntry, 'wb') as f:
            f.write(NewUPLEntry)

def AddSection64(UniversalPayloadEntry, AddSectionName, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, StringIndexNumber, FileBinary, Alignment):
    if False:
        print('Hello World!')
    with open(UniversalPayloadEntry, 'rb+') as f:
        UPLEntry = f.read()
        fFileBinary = open(FileBinary, 'rb')
        Binary_File = fFileBinary.read()
        (ElfHeaderOffset, SectionHeaderEntryNumber, StringIndexNumber, _, _, SectionHeaderEntrySize, _, _) = ElfHeaderParser(UPLEntry)
        SHentry = UPLEntry[ElfHeaderOffset:]
        elf_header = ElfHeader64(UPLEntry[:64])
        ElfHeaderSize = 64
        ElfHandPH = ElfHeaderSize + elf_header.e_phnum * elf_header.e_phentsize
        NewUPLEntry = UPLEntry[:ElfHandPH]
        NewUPLEntry = bytearray(NewUPLEntry)
        NewUPLEntrylen = []
        StringIndexValue = 0
        for Index in range(0, SectionHeaderEntryNumber):
            NewUPLEntrylen.append(len(NewUPLEntry))
            unpacked_SectionHeader = ElfSectionHeader64.unpack(SHentry[Index * SectionHeaderEntrySize:Index * SectionHeaderEntrySize + SectionHeaderEntrySize])
            if Index == 0:
                AlignmentIndex = 8
                if SectionHeaderEntryNumber > 2:
                    unpacked_NextSectionHeader = ElfSectionHeader64.unpack(SHentry[(Index + 1) * SectionHeaderEntrySize:(Index + 1) * SectionHeaderEntrySize + SectionHeaderEntrySize])
                NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_NextSectionHeader.sh_addralign)
            elif Index == SectionHeaderEntryNumber - 1:
                NewUPLEntry += UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                NewUPLEntry = SectionAlignment(NewUPLEntry, Alignment)
                LastUPLEntrylen = len(NewUPLEntry)
                NewUPLEntry += Binary_File
                AlignmentIndex = 8
                NewUPLEntry = SectionAlignment(NewUPLEntry, AlignmentIndex)
            elif Index == StringIndexNumber:
                StringIndex = UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                StringIndex = bytearray(StringIndex)
                StringIndexValue = len(StringIndex)
                AddSectionName = bytearray(AddSectionName, encoding='utf-8') + bytes('\x00', encoding='utf-8')
                StringIndex += AddSectionName
                NewUPLEntry += StringIndex
            elif Index > StringIndexNumber and Index < SectionHeaderEntryNumber - 1:
                NewUPLEntry += UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                unpacked_NextSectionHeader = ElfSectionHeader64.unpack(SHentry[(Index + 1) * SectionHeaderEntrySize:(Index + 1) * SectionHeaderEntrySize + SectionHeaderEntrySize])
                NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_NextSectionHeader.sh_addralign)
            else:
                NewUPLEntry += UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                if Index < SectionHeaderEntryNumber - 1:
                    unpacked_NextSectionHeader = ElfSectionHeader64.unpack(SHentry[(Index + 1) * SectionHeaderEntrySize:(Index + 1) * SectionHeaderEntrySize + SectionHeaderEntrySize])
                    NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_NextSectionHeader.sh_addralign)
        SectionHeaderOffset = len(NewUPLEntry)
        RemoveNameOffset = 0
        for Number in range(0, SectionHeaderEntryNumber):
            NewSHentry = AddSectionHeader64(SHentry, NewUPLEntrylen[Number], SectionHeaderEntrySize, Number, RemoveNameOffset, AddSectionName, StringIndexNumber)
            NewUPLEntry += NewSHentry
        NewUPLEntry += bytearray(AddNewSectionEntry64(LastUPLEntrylen, StringIndexValue, len(Binary_File), Alignment))
        elf_header.e_shoff = SectionHeaderOffset
        elf_header.e_shnum += 1
        elf_header = elf_header.pack()
        UPLEntryBin = elf_header + NewUPLEntry[64:]
        PHSegmentName = '.text'
        (_, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, _, StringIndexNumber, _) = FindSection(UPLEntryBin, PHSegmentName)
        UPLEntryBin = ModifyPHSegmentOffset64(UPLEntryBin, ElfHeaderOffset, PHSegmentName)
        PHSegmentName = '.dynamic'
        (_, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, _, StringIndexNumber, _) = FindSection(UPLEntryBin, PHSegmentName)
        UPLEntryBin = ModifyPHSegmentOffset64(UPLEntryBin, ElfHeaderOffset, PHSegmentName)
        PHSegmentName = '.data'
        (_, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, _, StringIndexNumber, _) = FindSection(UPLEntryBin, PHSegmentName)
        UPLEntryBin = ModifyPHSegmentOffset64(UPLEntryBin, ElfHeaderOffset, PHSegmentName)
    fFileBinary.close()
    return UPLEntryBin

def AddSection32(UniversalPayloadEntry, AddSectionName, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, StringIndexNumber, FileBinary, Alignment):
    if False:
        while True:
            i = 10
    with open(UniversalPayloadEntry, 'rb+') as f:
        UPLEntry = f.read()
        fFileBinary = open(FileBinary, 'rb')
        Binary_File = fFileBinary.read()
        (ElfHeaderOffset, SectionHeaderEntryNumber, StringIndexNumber, _, _, SectionHeaderEntrySize, _, _) = ElfHeaderParser(UPLEntry)
        SHentry = UPLEntry[ElfHeaderOffset:]
        elf_header = ElfHeader32(UPLEntry[:52])
        ElfHeaderSize = 52
        ElfHandPH = ElfHeaderSize + elf_header.e_phnum * elf_header.e_phentsize
        NewUPLEntry = UPLEntry[:ElfHandPH]
        NewUPLEntry = bytearray(NewUPLEntry)
        NewUPLEntrylen = []
        StringIndexValue = 0
        for Index in range(0, SectionHeaderEntryNumber):
            NewUPLEntrylen.append(len(NewUPLEntry))
            unpacked_SectionHeader = ElfSectionHeader32.unpack(SHentry[Index * SectionHeaderEntrySize:Index * SectionHeaderEntrySize + SectionHeaderEntrySize])
            if Index == 0:
                AlignmentIndex = 8
                if SectionHeaderEntryNumber > 2:
                    unpacked_NextSectionHeader = ElfSectionHeader32.unpack(SHentry[(Index + 1) * SectionHeaderEntrySize:(Index + 1) * SectionHeaderEntrySize + SectionHeaderEntrySize])
                NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_NextSectionHeader.sh_addralign)
            elif Index == SectionHeaderEntryNumber - 1:
                NewUPLEntry += UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                NewUPLEntry = SectionAlignment(NewUPLEntry, Alignment)
                LastUPLEntrylen = len(NewUPLEntry)
                NewUPLEntry += Binary_File
                AlignmentIndex = 8
                NewUPLEntry = SectionAlignment(NewUPLEntry, AlignmentIndex)
            elif Index == StringIndexNumber:
                StringIndex = UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                StringIndex = bytearray(StringIndex)
                StringIndexValue = len(StringIndex)
                AddSectionName = bytearray(AddSectionName, encoding='utf-8') + bytes('\x00', encoding='utf-8')
                StringIndex += AddSectionName
                NewUPLEntry += StringIndex
            elif Index > StringIndexNumber and Index < SectionHeaderEntryNumber - 1:
                NewUPLEntry += UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                unpacked_NextSectionHeader = ElfSectionHeader32.unpack(SHentry[(Index + 1) * SectionHeaderEntrySize:(Index + 1) * SectionHeaderEntrySize + SectionHeaderEntrySize])
                NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_NextSectionHeader.sh_addralign)
            else:
                NewUPLEntry += UPLEntry[unpacked_SectionHeader.sh_offset:unpacked_SectionHeader.sh_offset + unpacked_SectionHeader.sh_size]
                if Index < SectionHeaderEntryNumber - 1:
                    unpacked_NextSectionHeader = ElfSectionHeader32.unpack(SHentry[(Index + 1) * SectionHeaderEntrySize:(Index + 1) * SectionHeaderEntrySize + SectionHeaderEntrySize])
                    NewUPLEntry = SectionAlignment(NewUPLEntry, unpacked_NextSectionHeader.sh_addralign)
        SectionHeaderOffset = len(NewUPLEntry)
        RemoveNameOffset = 0
        for Number in range(0, SectionHeaderEntryNumber):
            NewSHentry = AddSectionHeader32(SHentry, NewUPLEntrylen[Number], SectionHeaderEntrySize, Number, RemoveNameOffset, AddSectionName, StringIndexNumber)
            NewUPLEntry += NewSHentry
        NewUPLEntry += bytearray(AddNewSectionEntry32(LastUPLEntrylen, StringIndexValue, len(Binary_File), Alignment))
        elf_header.e_shoff = SectionHeaderOffset
        elf_header.e_shnum += 1
        PHTableSize = elf_header.e_phentsize
        elf_header = elf_header.pack()
        UPLEntryBin = elf_header + NewUPLEntry[52:]
        PHSegmentName = '.text'
        (_, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, _, StringIndexNumber, _) = FindSection(UPLEntryBin, PHSegmentName)
        UPLEntryBin = ModifyPHSegmentOffset32(UPLEntryBin, ElfHeaderOffset, PHSegmentName)
        PHSegmentName = '.data'
        (_, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, _, StringIndexNumber, _) = FindSection(UPLEntryBin, PHSegmentName)
        UPLEntryBin = ModifyPHSegmentOffset32(UPLEntryBin, ElfHeaderOffset, PHSegmentName)
    fFileBinary.close()
    return UPLEntryBin

def ReplaceFv(UniversalPayloadEntry, FileBinary, AddSectionName, Alignment=16):
    if False:
        for i in range(10):
            print('nop')
    with open(UniversalPayloadEntry, 'rb+') as f:
        UPLEntry = f.read()
        (SectionNameOffset, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, _, StringIndexNumber, EI_CLASS) = FindSection(UPLEntry, AddSectionName)
    if EI_CLASS == 2:
        if SectionNameOffset != -1:
            RemoveSection64(UniversalPayloadEntry, AddSectionName)
        NewUPLEntry = AddSection64(UniversalPayloadEntry, AddSectionName, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, StringIndexNumber, FileBinary, Alignment)
    else:
        if SectionNameOffset != -1:
            RemoveSection32(UniversalPayloadEntry, AddSectionName)
        NewUPLEntry = AddSection32(UniversalPayloadEntry, AddSectionName, ElfHeaderOffset, SectionHeaderEntrySize, SectionHeaderEntryNumber, StringIndexNumber, FileBinary, Alignment)
    with open(UniversalPayloadEntry, 'wb') as f:
        f.write(NewUPLEntry)
    return 0