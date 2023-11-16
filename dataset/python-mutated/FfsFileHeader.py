from struct import *
from ctypes import *
from FirmwareStorageFormat.Common import *
EFI_FFS_FILE_HEADER_LEN = 24
EFI_FFS_FILE_HEADER2_LEN = 32

class CHECK_SUM(Structure):
    _pack_ = 1
    _fields_ = [('Header', c_uint8), ('File', c_uint8)]

class EFI_FFS_INTEGRITY_CHECK(Union):
    _pack_ = 1
    _fields_ = [('Checksum', CHECK_SUM), ('Checksum16', c_uint16)]

class EFI_FFS_FILE_HEADER(Structure):
    _pack_ = 1
    _fields_ = [('Name', GUID), ('IntegrityCheck', EFI_FFS_INTEGRITY_CHECK), ('Type', c_uint8), ('Attributes', c_uint8), ('Size', ARRAY(c_uint8, 3)), ('State', c_uint8)]

    @property
    def FFS_FILE_SIZE(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.Size[0] | self.Size[1] << 8 | self.Size[2] << 16

    @property
    def HeaderLength(self) -> int:
        if False:
            print('Hello World!')
        return 24

class EFI_FFS_FILE_HEADER2(Structure):
    _pack_ = 1
    _fields_ = [('Name', GUID), ('IntegrityCheck', EFI_FFS_INTEGRITY_CHECK), ('Type', c_uint8), ('Attributes', c_uint8), ('Size', ARRAY(c_uint8, 3)), ('State', c_uint8), ('ExtendedSize', c_uint64)]

    @property
    def FFS_FILE_SIZE(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.ExtendedSize

    @property
    def HeaderLength(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 32