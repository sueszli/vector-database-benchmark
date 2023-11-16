from struct import *
from ctypes import *
from FirmwareStorageFormat.Common import *
EFI_COMMON_SECTION_HEADER_LEN = 4
EFI_COMMON_SECTION_HEADER2_LEN = 8

class EFI_COMMON_SECTION_HEADER(Structure):
    _pack_ = 1
    _fields_ = [('Size', ARRAY(c_uint8, 3)), ('Type', c_uint8)]

    @property
    def SECTION_SIZE(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.Size[0] | self.Size[1] << 8 | self.Size[2] << 16

    def Common_Header_Size(self) -> int:
        if False:
            print('Hello World!')
        return 4

class EFI_COMMON_SECTION_HEADER2(Structure):
    _pack_ = 1
    _fields_ = [('Size', ARRAY(c_uint8, 3)), ('Type', c_uint8), ('ExtendedSize', c_uint32)]

    @property
    def SECTION_SIZE(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.ExtendedSize

    def Common_Header_Size(self) -> int:
        if False:
            print('Hello World!')
        return 8

class EFI_COMPRESSION_SECTION(Structure):
    _pack_ = 1
    _fields_ = [('UncompressedLength', c_uint32), ('CompressionType', c_uint8)]

    def ExtHeaderSize(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 5

class EFI_FREEFORM_SUBTYPE_GUID_SECTION(Structure):
    _pack_ = 1
    _fields_ = [('SubTypeGuid', GUID)]

    def ExtHeaderSize(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 16

class EFI_GUID_DEFINED_SECTION(Structure):
    _pack_ = 1
    _fields_ = [('SectionDefinitionGuid', GUID), ('DataOffset', c_uint16), ('Attributes', c_uint16)]

    def ExtHeaderSize(self) -> int:
        if False:
            print('Hello World!')
        return 20

def Get_USER_INTERFACE_Header(nums: int):
    if False:
        i = 10
        return i + 15

    class EFI_SECTION_USER_INTERFACE(Structure):
        _pack_ = 1
        _fields_ = [('FileNameString', ARRAY(c_uint16, nums))]

        def ExtHeaderSize(self) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return 2 * nums

        def GetUiString(self) -> str:
            if False:
                for i in range(10):
                    print('nop')
            UiString = ''
            for i in range(nums):
                if self.FileNameString[i]:
                    UiString += chr(self.FileNameString[i])
            return UiString
    return EFI_SECTION_USER_INTERFACE

def Get_VERSION_Header(nums: int):
    if False:
        for i in range(10):
            print('nop')

    class EFI_SECTION_VERSION(Structure):
        _pack_ = 1
        _fields_ = [('BuildNumber', c_uint16), ('VersionString', ARRAY(c_uint16, nums))]

        def ExtHeaderSize(self) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return 2 * (nums + 1)

        def GetVersionString(self) -> str:
            if False:
                print('Hello World!')
            VersionString = ''
            for i in range(nums):
                if self.VersionString[i]:
                    VersionString += chr(self.VersionString[i])
            return VersionString
    return EFI_SECTION_VERSION