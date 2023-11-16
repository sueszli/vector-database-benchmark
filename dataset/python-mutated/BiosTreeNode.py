from FirmwareStorageFormat.FvHeader import *
from FirmwareStorageFormat.FfsFileHeader import *
from FirmwareStorageFormat.SectionHeader import *
from FirmwareStorageFormat.Common import *
from utils.FmmtLogger import FmmtLogger as logger
import uuid
SectionHeaderType = {1: 'EFI_COMPRESSION_SECTION', 2: 'EFI_GUID_DEFINED_SECTION', 3: 'EFI_SECTION_DISPOSABLE', 16: 'EFI_SECTION_PE32', 17: 'EFI_SECTION_PIC', 18: 'EFI_SECTION_TE', 19: 'EFI_SECTION_DXE_DEPEX', 20: 'EFI_SECTION_VERSION', 21: 'EFI_SECTION_USER_INTERFACE', 22: 'EFI_SECTION_COMPATIBILITY16', 23: 'EFI_SECTION_FIRMWARE_VOLUME_IMAGE', 24: 'EFI_FREEFORM_SUBTYPE_GUID_SECTION', 25: 'EFI_SECTION_RAW', 27: 'EFI_SECTION_PEI_DEPEX', 28: 'EFI_SECTION_MM_DEPEX'}
HeaderType = [1, 2, 20, 21, 24]

class BinaryNode:

    def __init__(self, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.Size = 0
        self.Name = 'BINARY' + str(name)
        self.HOffset = 0
        self.Data = b''

class FvNode:

    def __init__(self, name, buffer: bytes) -> None:
        if False:
            print('Hello World!')
        self.Header = EFI_FIRMWARE_VOLUME_HEADER.from_buffer_copy(buffer)
        Map_num = (self.Header.HeaderLength - 56) // 8
        self.Header = Refine_FV_Header(Map_num).from_buffer_copy(buffer)
        self.FvId = 'FV' + str(name)
        self.Name = 'FV' + str(name)
        if self.Header.ExtHeaderOffset:
            self.ExtHeader = EFI_FIRMWARE_VOLUME_EXT_HEADER.from_buffer_copy(buffer[self.Header.ExtHeaderOffset:])
            self.Name = uuid.UUID(bytes_le=struct2stream(self.ExtHeader.FvName))
            self.ExtEntryOffset = self.Header.ExtHeaderOffset + 20
            if self.ExtHeader.ExtHeaderSize != 20:
                self.ExtEntryExist = 1
                self.ExtEntry = EFI_FIRMWARE_VOLUME_EXT_ENTRY.from_buffer_copy(buffer[self.ExtEntryOffset:])
                self.ExtTypeExist = 1
                if self.ExtEntry.ExtEntryType == 1:
                    nums = (self.ExtEntry.ExtEntrySize - 8) // 16
                    self.ExtEntry = Refine_FV_EXT_ENTRY_OEM_TYPE_Header(nums).from_buffer_copy(buffer[self.ExtEntryOffset:])
                elif self.ExtEntry.ExtEntryType == 2:
                    nums = self.ExtEntry.ExtEntrySize - 20
                    self.ExtEntry = Refine_FV_EXT_ENTRY_GUID_TYPE_Header(nums).from_buffer_copy(buffer[self.ExtEntryOffset:])
                elif self.ExtEntry.ExtEntryType == 3:
                    self.ExtEntry = EFI_FIRMWARE_VOLUME_EXT_ENTRY_USED_SIZE_TYPE.from_buffer_copy(buffer[self.ExtEntryOffset:])
                else:
                    self.ExtTypeExist = 0
            else:
                self.ExtEntryExist = 0
        self.Size = self.Header.FvLength
        self.HeaderLength = self.Header.HeaderLength
        self.HOffset = 0
        self.DOffset = 0
        self.ROffset = 0
        self.Data = b''
        if self.Header.Signature != 1213613663:
            logger.error('Invalid Fv Header! Fv {} signature {} is not "_FVH".'.format(struct2stream(self.Header), self.Header.Signature))
            raise Exception('Process Failed: Fv Header Signature!')
        self.PadData = b''
        self.Free_Space = 0
        self.ModCheckSum()

    def ModCheckSum(self) -> None:
        if False:
            i = 10
            return i + 15
        Header = struct2stream(self.Header)[::-1]
        Size = self.HeaderLength // 2
        Sum = 0
        for i in range(Size):
            Sum += int(Header[i * 2:i * 2 + 2].hex(), 16)
        if Sum & 65535:
            self.Header.Checksum = 65536 - (Sum - self.Header.Checksum) % 65536

    def ModFvExt(self) -> None:
        if False:
            return 10
        if self.Header.ExtHeaderOffset and self.ExtEntryExist and self.ExtTypeExist and (self.ExtEntry.Hdr.ExtEntryType == 3):
            self.ExtEntry.UsedSize = self.Header.FvLength - self.Free_Space

    def ModFvSize(self) -> None:
        if False:
            i = 10
            return i + 15
        BlockMapNum = len(self.Header.BlockMap)
        for i in range(BlockMapNum):
            if self.Header.BlockMap[i].Length:
                self.Header.BlockMap[i].NumBlocks = self.Header.FvLength // self.Header.BlockMap[i].Length

    def ModExtHeaderData(self) -> None:
        if False:
            while True:
                i = 10
        if self.Header.ExtHeaderOffset:
            ExtHeaderData = struct2stream(self.ExtHeader)
            ExtHeaderDataOffset = self.Header.ExtHeaderOffset - self.HeaderLength
            self.Data = self.Data[:ExtHeaderDataOffset] + ExtHeaderData + self.Data[ExtHeaderDataOffset + 20:]
        if self.Header.ExtHeaderOffset and self.ExtEntryExist:
            ExtHeaderEntryData = struct2stream(self.ExtEntry)
            ExtHeaderEntryDataOffset = self.Header.ExtHeaderOffset + 20 - self.HeaderLength
            self.Data = self.Data[:ExtHeaderEntryDataOffset] + ExtHeaderEntryData + self.Data[ExtHeaderEntryDataOffset + len(ExtHeaderEntryData):]

class FfsNode:

    def __init__(self, buffer: bytes) -> None:
        if False:
            print('Hello World!')
        self.Header = EFI_FFS_FILE_HEADER.from_buffer_copy(buffer)
        if self.Header.FFS_FILE_SIZE != 0 and self.Header.Attributes != 255 and (self.Header.Attributes & 1 == 1):
            logger.error('Error Ffs Header! Ffs {} Header Size and Attributes is not matched!'.format(uuid.UUID(bytes_le=struct2stream(self.Header.Name))))
            raise Exception('Process Failed: Error Ffs Header!')
        if self.Header.FFS_FILE_SIZE == 0 and self.Header.Attributes & 1 == 1:
            self.Header = EFI_FFS_FILE_HEADER2.from_buffer_copy(buffer)
        self.Name = uuid.UUID(bytes_le=struct2stream(self.Header.Name))
        self.UiName = b''
        self.Version = b''
        self.Size = self.Header.FFS_FILE_SIZE
        self.HeaderLength = self.Header.HeaderLength
        self.HOffset = 0
        self.DOffset = 0
        self.ROffset = 0
        self.Data = b''
        self.PadData = b''
        self.SectionMaxAlignment = SECTION_COMMON_ALIGNMENT

    def ModCheckSum(self) -> None:
        if False:
            while True:
                i = 10
        HeaderData = struct2stream(self.Header)
        HeaderSum = 0
        for item in HeaderData:
            HeaderSum += item
        HeaderSum -= self.Header.State
        HeaderSum -= self.Header.IntegrityCheck.Checksum.File
        if HeaderSum & 255:
            Header = self.Header.IntegrityCheck.Checksum.Header + 256 - HeaderSum % 256
            self.Header.IntegrityCheck.Checksum.Header = Header % 256

class SectionNode:

    def __init__(self, buffer: bytes) -> None:
        if False:
            return 10
        if buffer[0:3] != b'\xff\xff\xff':
            self.Header = EFI_COMMON_SECTION_HEADER.from_buffer_copy(buffer)
        else:
            self.Header = EFI_COMMON_SECTION_HEADER2.from_buffer_copy(buffer)
        if self.Header.Type in SectionHeaderType:
            self.Name = SectionHeaderType[self.Header.Type]
        elif self.Header.Type == 0:
            self.Name = 'EFI_SECTION_ALL'
        else:
            self.Name = 'SECTION'
        if self.Header.Type in HeaderType:
            self.ExtHeader = self.GetExtHeader(self.Header.Type, buffer[self.Header.Common_Header_Size():], self.Header.SECTION_SIZE - self.Header.Common_Header_Size())
            self.HeaderLength = self.Header.Common_Header_Size() + self.ExtHeader.ExtHeaderSize()
        else:
            self.ExtHeader = None
            self.HeaderLength = self.Header.Common_Header_Size()
        self.Size = self.Header.SECTION_SIZE
        self.Type = self.Header.Type
        self.HOffset = 0
        self.DOffset = 0
        self.ROffset = 0
        self.Data = b''
        self.OriData = b''
        self.OriHeader = b''
        self.PadData = b''
        self.IsPadSection = False
        self.SectionMaxAlignment = SECTION_COMMON_ALIGNMENT

    def GetExtHeader(self, Type: int, buffer: bytes, nums: int=0) -> None:
        if False:
            return 10
        if Type == 1:
            return EFI_COMPRESSION_SECTION.from_buffer_copy(buffer)
        elif Type == 2:
            return EFI_GUID_DEFINED_SECTION.from_buffer_copy(buffer)
        elif Type == 20:
            return Get_VERSION_Header((nums - 2) // 2).from_buffer_copy(buffer)
        elif Type == 21:
            return Get_USER_INTERFACE_Header(nums // 2).from_buffer_copy(buffer)
        elif Type == 24:
            return EFI_FREEFORM_SUBTYPE_GUID_SECTION.from_buffer_copy(buffer)

class FreeSpaceNode:

    def __init__(self, buffer: bytes) -> None:
        if False:
            print('Hello World!')
        self.Name = 'Free_Space'
        self.Data = buffer
        self.Size = len(buffer)
        self.HOffset = 0
        self.DOffset = 0
        self.ROffset = 0
        self.PadData = b''