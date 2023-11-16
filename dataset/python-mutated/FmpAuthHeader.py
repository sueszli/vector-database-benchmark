"""
FmpAuthHeader
"""
import struct
import uuid

class FmpAuthHeaderClass(object):
    _StructFormat = '<QIHH16s'
    _StructSize = struct.calcsize(_StructFormat)
    _MonotonicCountFormat = '<Q'
    _MonotonicCountSize = struct.calcsize(_MonotonicCountFormat)
    _StructAuthInfoFormat = '<IHH16s'
    _StructAuthInfoSize = struct.calcsize(_StructAuthInfoFormat)
    _WIN_CERT_REVISION = 512
    _WIN_CERT_TYPE_EFI_GUID = 3825
    _EFI_CERT_TYPE_PKCS7_GUID = uuid.UUID('4aafd29d-68df-49ee-8aa9-347d375665a7')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._Valid = False
        self.MonotonicCount = 0
        self.dwLength = self._StructAuthInfoSize
        self.wRevision = self._WIN_CERT_REVISION
        self.wCertificateType = self._WIN_CERT_TYPE_EFI_GUID
        self.CertType = self._EFI_CERT_TYPE_PKCS7_GUID
        self.CertData = b''
        self.Payload = b''

    def Encode(self):
        if False:
            print('Hello World!')
        if self.wRevision != self._WIN_CERT_REVISION:
            raise ValueError
        if self.wCertificateType != self._WIN_CERT_TYPE_EFI_GUID:
            raise ValueError
        if self.CertType != self._EFI_CERT_TYPE_PKCS7_GUID:
            raise ValueError
        self.dwLength = self._StructAuthInfoSize + len(self.CertData)
        FmpAuthHeader = struct.pack(self._StructFormat, self.MonotonicCount, self.dwLength, self.wRevision, self.wCertificateType, self.CertType.bytes_le)
        self._Valid = True
        return FmpAuthHeader + self.CertData + self.Payload

    def Decode(self, Buffer):
        if False:
            i = 10
            return i + 15
        if len(Buffer) < self._StructSize:
            raise ValueError
        (MonotonicCount, dwLength, wRevision, wCertificateType, CertType) = struct.unpack(self._StructFormat, Buffer[0:self._StructSize])
        if dwLength < self._StructAuthInfoSize:
            raise ValueError
        if wRevision != self._WIN_CERT_REVISION:
            raise ValueError
        if wCertificateType != self._WIN_CERT_TYPE_EFI_GUID:
            raise ValueError
        if CertType != self._EFI_CERT_TYPE_PKCS7_GUID.bytes_le:
            raise ValueError
        self.MonotonicCount = MonotonicCount
        self.dwLength = dwLength
        self.wRevision = wRevision
        self.wCertificateType = wCertificateType
        self.CertType = uuid.UUID(bytes_le=CertType)
        self.CertData = Buffer[self._StructSize:self._MonotonicCountSize + self.dwLength]
        self.Payload = Buffer[self._MonotonicCountSize + self.dwLength:]
        self._Valid = True
        return self.Payload

    def IsSigned(self, Buffer):
        if False:
            for i in range(10):
                print('nop')
        if len(Buffer) < self._StructSize:
            return False
        (MonotonicCount, dwLength, wRevision, wCertificateType, CertType) = struct.unpack(self._StructFormat, Buffer[0:self._StructSize])
        if CertType != self._EFI_CERT_TYPE_PKCS7_GUID.bytes_le:
            return False
        return True

    def DumpInfo(self):
        if False:
            print('Hello World!')
        if not self._Valid:
            raise ValueError
        print('EFI_FIRMWARE_IMAGE_AUTHENTICATION.MonotonicCount                = {MonotonicCount:016X}'.format(MonotonicCount=self.MonotonicCount))
        print('EFI_FIRMWARE_IMAGE_AUTHENTICATION.AuthInfo.Hdr.dwLength         = {dwLength:08X}'.format(dwLength=self.dwLength))
        print('EFI_FIRMWARE_IMAGE_AUTHENTICATION.AuthInfo.Hdr.wRevision        = {wRevision:04X}'.format(wRevision=self.wRevision))
        print('EFI_FIRMWARE_IMAGE_AUTHENTICATION.AuthInfo.Hdr.wCertificateType = {wCertificateType:04X}'.format(wCertificateType=self.wCertificateType))
        print('EFI_FIRMWARE_IMAGE_AUTHENTICATION.AuthInfo.CertType             = {Guid}'.format(Guid=str(self.CertType).upper()))
        print('sizeof (EFI_FIRMWARE_IMAGE_AUTHENTICATION.AuthInfo.CertData)    = {Size:08X}'.format(Size=len(self.CertData)))
        print('sizeof (Payload)                                                = {Size:08X}'.format(Size=len(self.Payload)))