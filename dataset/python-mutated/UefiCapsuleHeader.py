"""
UefiCapsuleHeader
"""
import struct
import uuid

class UefiCapsuleHeaderClass(object):
    _StructFormat = '<16sIIII'
    _StructSize = struct.calcsize(_StructFormat)
    EFI_FIRMWARE_MANAGEMENT_CAPSULE_ID_GUID = uuid.UUID('6DCBD5ED-E82D-4C44-BDA1-7194199AD92A')
    _CAPSULE_FLAGS_PERSIST_ACROSS_RESET = 65536
    _CAPSULE_FLAGS_POPULATE_SYSTEM_TABLE = 131072
    _CAPSULE_FLAGS_INITIATE_RESET = 262144

    def __init__(self):
        if False:
            while True:
                i = 10
        self._Valid = False
        self.CapsuleGuid = self.EFI_FIRMWARE_MANAGEMENT_CAPSULE_ID_GUID
        self.HeaderSize = self._StructSize
        self.OemFlags = 0
        self.PersistAcrossReset = False
        self.PopulateSystemTable = False
        self.InitiateReset = False
        self.CapsuleImageSize = self.HeaderSize
        self.Payload = b''

    def Encode(self):
        if False:
            i = 10
            return i + 15
        Flags = self.OemFlags
        if self.PersistAcrossReset:
            Flags = Flags | self._CAPSULE_FLAGS_PERSIST_ACROSS_RESET
        if self.PopulateSystemTable:
            Flags = Flags | self._CAPSULE_FLAGS_POPULATE_SYSTEM_TABLE
        if self.InitiateReset:
            Flags = Flags | self._CAPSULE_FLAGS_INITIATE_RESET
        self.CapsuleImageSize = self.HeaderSize + len(self.Payload)
        UefiCapsuleHeader = struct.pack(self._StructFormat, self.CapsuleGuid.bytes_le, self.HeaderSize, Flags, self.CapsuleImageSize, 0)
        self._Valid = True
        return UefiCapsuleHeader + self.Payload

    def Decode(self, Buffer):
        if False:
            for i in range(10):
                print('nop')
        if len(Buffer) < self._StructSize:
            raise ValueError
        (CapsuleGuid, HeaderSize, Flags, CapsuleImageSize, Reserved) = struct.unpack(self._StructFormat, Buffer[0:self._StructSize])
        if HeaderSize < self._StructSize:
            raise ValueError
        if CapsuleImageSize != len(Buffer):
            raise ValueError
        self.CapsuleGuid = uuid.UUID(bytes_le=CapsuleGuid)
        self.HeaderSize = HeaderSize
        self.OemFlags = Flags & 65535
        self.PersistAcrossReset = Flags & self._CAPSULE_FLAGS_PERSIST_ACROSS_RESET != 0
        self.PopulateSystemTable = Flags & self._CAPSULE_FLAGS_POPULATE_SYSTEM_TABLE != 0
        self.InitiateReset = Flags & self._CAPSULE_FLAGS_INITIATE_RESET != 0
        self.CapsuleImageSize = CapsuleImageSize
        self.Payload = Buffer[self.HeaderSize:]
        self._Valid = True
        return self.Payload

    def DumpInfo(self):
        if False:
            i = 10
            return i + 15
        if not self._Valid:
            raise ValueError
        Flags = self.OemFlags
        if self.PersistAcrossReset:
            Flags = Flags | self._CAPSULE_FLAGS_PERSIST_ACROSS_RESET
        if self.PopulateSystemTable:
            Flags = Flags | self._CAPSULE_FLAGS_POPULATE_SYSTEM_TABLE
        if self.InitiateReset:
            Flags = Flags | self._CAPSULE_FLAGS_INITIATE_RESET
        print('EFI_CAPSULE_HEADER.CapsuleGuid      = {Guid}'.format(Guid=str(self.CapsuleGuid).upper()))
        print('EFI_CAPSULE_HEADER.HeaderSize       = {Size:08X}'.format(Size=self.HeaderSize))
        print('EFI_CAPSULE_HEADER.Flags            = {Flags:08X}'.format(Flags=Flags))
        print('  OEM Flags                         = {Flags:04X}'.format(Flags=self.OemFlags))
        if self.PersistAcrossReset:
            print('  CAPSULE_FLAGS_PERSIST_ACROSS_RESET')
        if self.PopulateSystemTable:
            print('  CAPSULE_FLAGS_POPULATE_SYSTEM_TABLE')
        if self.InitiateReset:
            print('  CAPSULE_FLAGS_INITIATE_RESET')
        print('EFI_CAPSULE_HEADER.CapsuleImageSize = {Size:08X}'.format(Size=self.CapsuleImageSize))
        print('sizeof (Payload)                    = {Size:08X}'.format(Size=len(self.Payload)))