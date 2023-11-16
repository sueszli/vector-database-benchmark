"""
SMB (Server Message Block), also known as CIFS - version 2
"""
import struct
from scapy.config import conf
from scapy.error import log_runtime
from scapy.packet import Packet, bind_layers, bind_top_down
from scapy.fields import ByteEnumField, ByteField, ConditionalField, FieldLenField, FieldListField, FlagsField, IntEnumField, IntField, LEIntField, LEIntEnumField, LELongField, LEShortEnumField, LEShortField, MultipleTypeField, PadField, PacketField, PacketLenField, PacketListField, ReversePadField, ShortEnumField, ShortField, StrFieldUtf16, StrFixedLenField, StrLenFieldUtf16, StrLenField, StrNullFieldUtf16, UTCTimeField, UUIDField, XLEIntField, XLELongField, XLEShortField, XStrLenField, XStrFixedLenField
from scapy.layers.netbios import NBTSession
from scapy.layers.gssapi import GSSAPI_BLOB
from scapy.layers.ntlm import _NTLMPayloadField, _NTLMPayloadPacket
SMB_DIALECTS = {514: 'SMB 2.002', 528: 'SMB 2.1', 767: 'SMB 2.???', 768: 'SMB 3.0', 770: 'SMB 3.0.2', 785: 'SMB 3.1.1'}
STATUS_ERREF = {0: 'STATUS_SUCCESS', 259: 'STATUS_PENDING', 268: 'STATUS_NOTIFY_ENUM_DIR', 3221225494: 'STATUS_MORE_PROCESSING_REQUIRED', 3221225506: 'STATUS_ACCESS_DENIED', 3221225524: 'STATUS_OBJECT_NAME_NOT_FOUND', 3221225626: 'STATUS_INSUFFICIENT_RESOURCES', 3221225760: 'STATUS_CANCELLED', 3221225768: 'STATUS_FILE_CLOSED', 3221225485: 'STATUS_INVALID_PARAMETER', 3221225487: 'STATUS_NO_SUCH_FILE', 3221225659: 'STATUS_NOT_SUPPORTED', 3221225884: 'STATUS_FS_DRIVER_REQUIRED', 3221226021: 'STATUS_NOT_FOUND', 2147483653: 'STATUS_BUFFER_OVERFLOW', 2147483654: 'STATUS_NO_MORE_FILES'}
SMB2_COM = {0: 'SMB2_NEGOTIATE', 1: 'SMB2_SESSION_SETUP', 2: 'SMB2_LOGOFF', 3: 'SMB2_TREE_CONNECT', 4: 'SMB2_TREE_DISCONNECT', 5: 'SMB2_CREATE', 6: 'SMB2_CLOSE', 7: 'SMB2_FLUSH', 8: 'SMB2_READ', 9: 'SMB2_WRITE', 10: 'SMB2_LOCK', 11: 'SMB2_IOCTL', 12: 'SMB2_CANCEL', 13: 'SMB2_ECHO', 14: 'SMB2_QUERY_DIRECTORY', 15: 'SMB2_CHANGE_NOTIFY', 16: 'SMB2_QUERY_INFO', 17: 'SMB2_SET_INFO', 18: 'SMB2_OPLOCK_BREAK'}
SMB2_NEGOTIATE_CONTEXT_TYPES = {1: 'SMB2_PREAUTH_INTEGRITY_CAPABILITIES', 2: 'SMB2_ENCRYPTION_CAPABILITIES', 3: 'SMB2_COMPRESSION_CAPABILITIES', 5: 'SMB2_NETNAME_NEGOTIATE_CONTEXT_ID', 6: 'SMB2_TRANSPORT_CAPABILITIES', 7: 'SMB2_RDMA_TRANSFORM_CAPABILITIES', 8: 'SMB2_SIGNING_CAPABILITIES'}
SMB2_CAPABILITIES = {1: 'DFS', 2: 'Leasing', 4: 'LargeMTU', 8: 'MultiChannel', 16: 'PersistentHandles', 32: 'DirectoryLeasing', 64: 'Encryption'}
SMB2_COMPRESSION_ALGORITHMS = {0: 'None', 1: 'LZNT1', 2: 'LZ77', 3: 'LZ77 + Huffman', 4: 'Pattern_V1'}
FileAttributes = {1: 'FILE_ATTRIBUTE_READONLY', 2: 'FILE_ATTRIBUTE_HIDDEN', 4: 'FILE_ATTRIBUTE_SYSTEM', 16: 'FILE_ATTRIBUTE_DIRECTORY', 32: 'FILE_ATTRIBUTE_ARCHIVE', 128: 'FILE_ATTRIBUTE_NORMAL', 256: 'FILE_ATTRIBUTE_TEMPORARY', 512: 'FILE_ATTRIBUTE_SPARSE_FILE', 1024: 'FILE_ATTRIBUTE_REPARSE_POINT', 2048: 'FILE_ATTRIBUTE_COMPRESSED', 4096: 'FILE_ATTRIBUTE_OFFLINE', 8192: 'FILE_ATTRIBUTE_NOT_CONTENT_INDEXED', 16384: 'FILE_ATTRIBUTE_ENCRYPTED', 32768: 'FILE_ATTRIBUTE_INTEGRITY_STREAM', 131072: 'FILE_ATTRIBUTE_NO_SCRUB_DATA', 262144: 'FILE_ATTRIBUTE_RECALL_ON_OPEN', 524288: 'FILE_ATTRIBUTE_PINNED', 1048576: 'FILE_ATTRIBUTE_UNPINNED', 4194304: 'FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS'}
FileInformationClasses = {1: 'FileDirectoryInformation', 2: 'FileFullDirectoryInformation', 3: 'FileBothDirectoryInformation', 5: 'FileStandardInformation', 6: 'FileInternalInformation', 7: 'FileEaInformation', 34: 'FileNetworkOpenInformation', 37: 'FileIdBothDirectoryInformation', 38: 'FileIdFullDirectoryInformation', 12: 'FileNamesInformation', 60: 'FileIdExtdDirectoryInformation'}

class FileEaInformation(Packet):
    fields_desc = [LEIntField('EaSize', 0)]

class FileNetworkOpenInformation(Packet):
    fields_desc = [UTCTimeField('CreationTime', None, fmt='<Q', epoch=[1601, 1, 1, 0, 0, 0], custom_scaling=10000000.0), UTCTimeField('LastAccessTime', None, fmt='<Q', epoch=[1601, 1, 1, 0, 0, 0], custom_scaling=10000000.0), UTCTimeField('LastWriteTime', None, fmt='<Q', epoch=[1601, 1, 1, 0, 0, 0], custom_scaling=10000000.0), UTCTimeField('ChangeTime', None, fmt='<Q', epoch=[1601, 1, 1, 0, 0, 0], custom_scaling=10000000.0), LELongField('AllocationSize', 4096), LELongField('EnfofFile', 4096), FlagsField('FileAttributes', 128, -32, FileAttributes), IntField('Reserved2', 0)]

class FILE_ID_BOTH_DIR_INFORMATION(Packet):
    fields_desc = [LEIntField('Next', None), LEIntField('FileIndex', 0)] + FileNetworkOpenInformation.fields_desc[:7] + [FieldLenField('FileNameLength', None, fmt='<I', length_of='FileName'), LEIntField('EaSize', 0), ByteField('ShortNameLength', 0), ByteField('Reserved1', 0), StrFixedLenField('ShortName', b'', length=24), LEShortField('Reserved2', 0), LELongField('FileId', 0), PadField(StrLenFieldUtf16('FileName', b'.', length_from=lambda pkt: pkt.FileNameLength), align=8)]

    def default_payload_class(self, s):
        if False:
            i = 10
            return i + 15
        return conf.padding_layer

class _NextPacketListField(PacketListField):

    def addfield(self, pkt, s, val):
        if False:
            for i in range(10):
                print('nop')
        res = b''
        for (i, v) in enumerate(val):
            x = self.i2m(pkt, v)
            if v.Next is None and i != len(val) - 1:
                x = struct.pack('<I', len(x)) + x[4:]
            res += x
        return s + res

class FileIdBothDirectoryInformation(Packet):
    fields_desc = [_NextPacketListField('files', [], FILE_ID_BOTH_DIR_INFORMATION)]

class FileInternalInformation(Packet):
    fields_desc = [LELongField('IndexNumber', 0)]

class FileStandardInformation(Packet):
    fields_desc = [LELongField('AllocationSize', 4096), LELongField('EndOfFile', 0), LEIntField('NumberOfLinks', 1), ByteField('DeletePending', 0), ByteField('Directory', 0), ShortField('Reserved', 0)]

class FileStreamInformation(Packet):
    fields_desc = [LEIntField('Next', 0), FieldLenField('StreamNameLength', None, length_of='StreamName', fmt='<I'), LELongField('StreamSize', 0), LELongField('StreamAllocationSize', 0), StrLenFieldUtf16('StreamName', b'::$DATA', length_from=lambda pkt: pkt.StreamNameLength)]

class SECURITY_DESCRIPTOR(Packet):
    fields_desc = [ByteField('Revision', 1), ByteField('Sbz1', 0), FlagsField('Control', 0, -16, ['SelfRelative', 'RMControlValid', 'SACLProtected', 'DACLProtected', 'SACLAutoInherited', 'DACLAutoInheriter', 'SACLComputer', 'DACLComputer', 'ServerSecurity', 'DACLTrusted', 'SACLDefaulted', 'SACLPresent', 'DACLDefaulted', 'DACLPresent', 'GroupDefaulted', 'OwnerDefaulted']), LEIntField('OffsetOwner', 0), LEIntField('OffsetGroup', 0), LEIntField('OffsetSacl', 0), LEIntField('OffsetDacl', 0), ConditionalField(XStrLenField('OwnerSid', b''), lambda pkt: pkt.OffsetOwner), ConditionalField(XStrLenField('GroupSid', b''), lambda pkt: pkt.OffsetGroup), ConditionalField(XStrLenField('Sacl', b''), lambda pkt: pkt.Control.SACLPresent), ConditionalField(XStrLenField('Dacl', b''), lambda pkt: pkt.Control.DACLPresent)]

class FileFsAttributeInformation(Packet):
    fields_desc = [FlagsField('FileSystemAttributes', 8388623, -32, {33554432: 'FILE_SUPPORTS_USN_JOURNAL', 16777216: 'FILE_SUPPORTS_OPEN_BY_FILE_ID', 8388608: 'FILE_SUPPORTS_EXTENDED_ATTRIBUTES', 4194304: 'FILE_SUPPORTS_HARD_LINKS', 2097152: 'FILE_SUPPORTS_TRANSACTIONS', 1048576: 'FILE_SEQUENTIAL_WRITE_ONCE', 524288: 'FILE_READ_ONLY_VOLUME', 262144: 'FILE_NAMED_STREAMS', 131072: 'FILE_SUPPORTS_ENCRYPTION', 65536: 'FILE_SUPPORTS_OBJECT_IDS', 32768: 'FILE_VOLUME_IS_COMPRESSED', 256: 'FILE_SUPPORTS_REMOTE_STORAGE', 128: 'FILE_SUPPORTS_REPARSE_POINTS', 64: 'FILE_SUPPORTS_SPARSE_FILES', 32: 'FILE_VOLUME_QUOTAS', 16: 'FILE_FILE_COMPRESSION', 8: 'FILE_PERSISTENT_ACLS', 4: 'FILE_UNICODE_ON_DISK', 2: 'FILE_CASE_PRESERVED_NAMES', 1: 'FILE_CASE_SENSITIVE_SEARCH', 67108864: 'FILE_SUPPORT_INTEGRITY_STREAMS', 134217728: 'FILE_SUPPORTS_BLOCK_REFCOUNTING', 268435456: 'FILE_SUPPORTS_SPARSE_VDL'}), LEIntField('MaximumComponentNameLength', 255), FieldLenField('FileSystemNameLength', None, length_of='FileSystemName', fmt='<I'), StrLenFieldUtf16('FileSystemName', b'NTFS', length_from=lambda pkt: pkt.FileSystemNameLength)]

class FileFsSizeInformation(Packet):
    fields_desc = [LELongField('TotalAllocationUnits', 10485760), LELongField('AvailableAllocationUnits', 1048576), LEIntField('SectorsPerAllocationUnit', 8), LEIntField('BytesPerSector', 512)]

class FileFsVolumeInformation(Packet):
    fields_desc = [UTCTimeField('VolumeCreationTime', None, fmt='<Q', epoch=[1601, 1, 1, 0, 0, 0], custom_scaling=10000000.0), LEIntField('VolumeSerialNumber', 0), LEIntField('VolumeLabelLength', 0), ByteField('SupportsObjects', 1), ByteField('Reserved', 0), StrNullFieldUtf16('VolumeLabel', b'C')]

def _SMB2_post_build(self, p, pay_offset, fields):
    if False:
        print('Hello World!')
    'Util function to build the offset and populate the lengths'
    for (field_name, offset) in fields.items():
        try:
            value = next((x[1] for x in self.fields['Buffer'] if x[0] == field_name))
            length = self.get_field('Buffer').fields_map[field_name].i2len(self, value)
        except StopIteration:
            length = 0
        i = 0
        r = lambda y: {2: 'H', 4: 'I', 8: 'Q'}[y]
        if self.getfieldval(field_name + 'BufferOffset') is None:
            sz = self.get_field(field_name + 'BufferOffset').sz
            p = p[:offset + i] + struct.pack('<%s' % r(sz), pay_offset) + p[offset + sz:]
            i += sz
        if self.getfieldval(field_name + 'Len') is None:
            sz = self.get_field(field_name + 'Len').sz
            p = p[:offset + i] + struct.pack('<%s' % r(sz), length) + p[offset + i + sz:]
            i += sz
        pay_offset += length
    return p

class SMB2_Header(Packet):
    name = 'SMB2 Header'
    fields_desc = [StrFixedLenField('Start', b'\xfeSMB', 4), LEShortField('StructureSize', 64), LEShortField('CreditCharge', 0), LEIntEnumField('Status', 0, STATUS_ERREF), LEShortEnumField('Command', 0, SMB2_COM), LEShortField('CreditsRequested', 0), FlagsField('Flags', 0, -32, {1: 'SMB2_FLAGS_SERVER_TO_REDIR', 2: 'SMB2_FLAGS_ASYNC_COMMAND', 4: 'SMB2_FLAGS_RELATED_OPERATIONS', 8: 'SMB2_FLAGS_SIGNED', 268435456: 'SMB2_FLAGS_DFS_OPERATIONS', 536870912: 'SMB2_FLAGS_REPLAY_OPERATION'}), XLEIntField('NextCommand', 0), LELongField('MID', 0), ConditionalField(LELongField('AsyncId', 0), lambda pkt: pkt.Flags.SMB2_FLAGS_ASYNC_COMMAND), ConditionalField(LEIntField('PID', 0), lambda pkt: not pkt.Flags.SMB2_FLAGS_ASYNC_COMMAND), ConditionalField(LEIntField('TID', 0), lambda pkt: not pkt.Flags.SMB2_FLAGS_ASYNC_COMMAND), LELongField('SessionId', 0), XStrFixedLenField('SecuritySignature', 0, length=16)]
    _SMB2_OK_RETURNCODES = (0, 3221225494, 2147483653, 3221225485, 268)

    def guess_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
            if self.Status not in SMB2_Header._SMB2_OK_RETURNCODES:
                return SMB2_Error_Response
        if self.Command == 0:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Negotiate_Protocol_Response
            return SMB2_Negotiate_Protocol_Request
        elif self.Command == 1:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Session_Setup_Response
            return SMB2_Session_Setup_Request
        elif self.Command == 2:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Session_Logoff_Response
            return SMB2_Session_Logoff_Request
        elif self.Command == 3:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Tree_Connect_Response
            return SMB2_Tree_Connect_Request
        elif self.Command == 4:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Tree_Disconnect_Response
            return SMB2_Tree_Disconnect_Request
        elif self.Command == 5:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Create_Response
            return SMB2_Create_Request
        elif self.Command == 6:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Close_Response
            return SMB2_Close_Request
        elif self.Command == 8:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Read_Response
            return SMB2_Read_Request
        elif self.Command == 9:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Write_Response
            return SMB2_Write_Request
        elif self.Command == 12:
            return SMB2_Cancel_Request
        elif self.Command == 14:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Query_Directory_Response
            return SMB2_Query_Directory_Request
        elif self.Command == 15:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Change_Notify_Response
            return SMB2_Change_Notify_Request
        elif self.Command == 16:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_Query_Info_Response
            return SMB2_Query_Info_Request
        elif self.Command == 11:
            if self.Flags.SMB2_FLAGS_SERVER_TO_REDIR:
                return SMB2_IOCTL_Response
            return SMB2_IOCTL_Request
        return super(SMB2_Header, self).guess_payload_class(payload)

    def sign(self, dialect, SigningSessionKey):
        if False:
            print('Hello World!')
        self.SecuritySignature = b'\x00' * 16
        s = bytes(self)
        if len(s) <= 64:
            log_runtime.warning('Cannot sign invalid SMB packet !')
            return s
        if dialect == 785:
            raise Exception('SMB 3.1.1 signing unimplemented')
        elif dialect in [768, 770]:
            from cryptography.hazmat.primitives import cmac
            from cryptography.hazmat.primitives.ciphers import algorithms
            c = cmac.CMAC(algorithms.AES(SigningSessionKey))
            c.update(s)
            sig = c.finalize()
        elif dialect in [528, 514]:
            from scapy.layers.tls.crypto.h_mac import Hmac_SHA256
            sig = Hmac_SHA256(SigningSessionKey).digest(s)
            sig = sig[:16]
        else:
            log_runtime.warning('Unknown SMB Version %s ! Cannot sign.' % dialect)
            sig = s[:-16] + b'\x00' * 16
        self.SecuritySignature = sig

class _SMB2_Payload(Packet):

    def do_dissect_payload(self, s):
        if False:
            while True:
                i = 10
        if self.underlayer and isinstance(self.underlayer, SMB2_Header):
            if self.underlayer.NextCommand:
                padlen = self.underlayer.NextCommand - (64 + len(self.raw_packet_cache))
                if padlen:
                    self.add_payload(conf.padding_layer(s[:padlen]))
                    s = s[padlen:]
        super(_SMB2_Payload, self).do_dissect_payload(s)

    def guess_payload_class(self, s):
        if False:
            print('Hello World!')
        if self.underlayer and isinstance(self.underlayer, SMB2_Header):
            if self.underlayer.NextCommand:
                return SMB2_Header
        return NBTSession

class SMB2_Error_Response(_SMB2_Payload):
    name = 'SMB2 Error Response'
    fields_desc = [XLEShortField('StructureSize', 9), ByteField('ErrorContextCount', 0), ByteField('Reserved', 0), FieldLenField('ByteCount', None, fmt='<I', length_of='ErrorData'), XStrLenField('ErrorData', b'', length_from=lambda pkt: pkt.ByteCount)]
bind_top_down(SMB2_Header, SMB2_Error_Response, Flags=1)

class SMB2_Negotiate_Context(Packet):
    name = 'SMB2 Negotiate Context'
    fields_desc = [LEShortEnumField('ContextType', 0, SMB2_NEGOTIATE_CONTEXT_TYPES), FieldLenField('DataLength', 0, fmt='<H', length_of='Data'), IntField('Reserved', 0)]

class SMB2_Negotiate_Protocol_Request(_SMB2_Payload):
    name = 'SMB2 Negotiate Protocol Request'
    fields_desc = [XLEShortField('StructureSize', 36), FieldLenField('DialectCount', None, fmt='<H', count_of='Dialects'), FlagsField('SecurityMode', 0, -16, {1: 'SMB2_NEGOTIATE_SIGNING_ENABLED', 2: 'SMB2_NEGOTIATE_SIGNING_REQUIRED'}), LEShortField('Reserved', 0), FlagsField('Capabilities', 0, -32, SMB2_CAPABILITIES), UUIDField('ClientGUID', 0, uuid_fmt=UUIDField.FORMAT_LE), XLEIntField('NegotiateContextOffset', 0), FieldLenField('NegotiateCount', None, fmt='<H', count_of='NegotiateContexts'), ShortField('Reserved2', 0), FieldListField('Dialects', [514], LEShortEnumField('', 0, SMB_DIALECTS), count_from=lambda pkt: pkt.DialectCount), ConditionalField(FieldListField('NegotiateContexts', [], ReversePadField(PacketField('Context', None, SMB2_Negotiate_Context), 8), count_from=lambda pkt: pkt.NegotiateCount), lambda x: 785 in x.Dialects)]
bind_top_down(SMB2_Header, SMB2_Negotiate_Protocol_Request, Command=0)

class SMB2_Preauth_Integrity_Capabilities(Packet):
    name = 'SMB2 Preauth Integrity Capabilities'
    fields_desc = [FieldLenField('HashAlgorithmCount', 1, fmt='<H', count_of='HashAlgorithms'), FieldLenField('SaltLength', 0, fmt='<H', length_of='Salt'), FieldListField('HashAlgorithms', [1], LEShortEnumField('', 0, {1: 'SHA-512'}), count_from=lambda pkt: pkt.HashAlgorithmCount), XStrLenField('Salt', '', length_from=lambda pkt: pkt.SaltLength)]

    def default_payload_class(self, payload):
        if False:
            return 10
        return conf.padding_layer
bind_layers(SMB2_Negotiate_Context, SMB2_Preauth_Integrity_Capabilities, ContextType=1)

class SMB2_Encryption_Capabilities(Packet):
    name = 'SMB2 Encryption Capabilities'
    fields_desc = [FieldLenField('CipherCount', 1, fmt='<H', count_of='Ciphers'), FieldListField('Ciphers', [1], LEShortEnumField('', 0, {1: 'AES-128-CCM', 2: 'AES-128-GCM'}), count_from=lambda pkt: pkt.CipherCount)]

    def default_payload_class(self, payload):
        if False:
            print('Hello World!')
        return conf.padding_layer
bind_layers(SMB2_Negotiate_Context, SMB2_Encryption_Capabilities, ContextType=2)

class SMB2_Compression_Capabilities(Packet):
    name = 'SMB2 Compression Capabilities'
    fields_desc = [FieldLenField('CompressionAlgorithmCount', 0, fmt='<H', count_of='CompressionAlgorithms'), ShortField('Padding', 0), IntEnumField('Flags', 0, {0: 'SMB2_COMPRESSION_CAPABILITIES_FLAG_NONE', 1: 'SMB2_COMPRESSION_CAPABILITIES_FLAG_CHAINED'}), FieldListField('CompressionAlgorithms', None, LEShortEnumField('', 0, SMB2_COMPRESSION_ALGORITHMS), count_from=lambda pkt: pkt.CompressionAlgorithmCount)]

    def default_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        return conf.padding_layer
bind_layers(SMB2_Negotiate_Context, SMB2_Compression_Capabilities, ContextType=3)

class SMB2_Netname_Negotiate_Context_ID(Packet):
    name = 'SMB2 Netname Negotiate Context ID'
    fields_desc = [StrFieldUtf16('NetName', '')]

    def default_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        return conf.padding_layer
bind_layers(SMB2_Negotiate_Context, SMB2_Netname_Negotiate_Context_ID, ContextType=5)

class SMB2_Transport_Capabilities(Packet):
    name = 'SMB2 Transport Capabilities'
    fields_desc = [FlagsField('Flags', 0, -32, {1: 'SMB2_ACCEPT_TRANSPORT_LEVEL_SECURITY'})]

    def default_payload_class(self, payload):
        if False:
            return 10
        return conf.padding_layer
bind_layers(SMB2_Negotiate_Context, SMB2_Transport_Capabilities, ContextType=6)

class SMB2_Negotiate_Protocol_Response(_SMB2_Payload):
    name = 'SMB2 Negotiate Protocol Response'
    fields_desc = [XLEShortField('StructureSize', 65), FlagsField('SecurityMode', 0, -16, {1: 'Signing Required', 2: 'Signing Enabled'}), LEShortEnumField('DialectRevision', 0, SMB_DIALECTS), FieldLenField('NegotiateCount', None, fmt='<H', count_of='NegotiateContexts'), UUIDField('GUID', 0, uuid_fmt=UUIDField.FORMAT_LE), FlagsField('Capabilities', 0, -32, SMB2_CAPABILITIES), LEIntField('MaxTransactionSize', 65536), LEIntField('MaxReadSize', 65536), LEIntField('MaxWriteSize', 65536), UTCTimeField('ServerTime', None, fmt='<Q', epoch=[1601, 1, 1, 0, 0, 0], custom_scaling=10000000.0), UTCTimeField('ServerStartTime', None, fmt='<Q', epoch=[1601, 1, 1, 0, 0, 0], custom_scaling=10000000.0), FieldLenField('SecurityBlobOffset', None, fmt='<H', length_of='SecurityBlobPad', adjust=lambda pkt, x: x + 128), FieldLenField('SecurityBlobLength', None, fmt='<H', length_of='SecurityBlob'), XLEIntField('NegotiateContextOffset', 0), XStrLenField('SecurityBlobPad', '', length_from=lambda pkt: pkt.SecurityBlobOffset - 128), PacketLenField('SecurityBlob', None, GSSAPI_BLOB, length_from=lambda x: x.SecurityBlobLength), ConditionalField(FieldListField('NegotiateContexts', [], ReversePadField(PacketField('Context', None, SMB2_Negotiate_Context), 8), count_from=lambda pkt: pkt.NegotiateCount), lambda x: x.DialectRevision == 785)]
bind_top_down(SMB2_Header, SMB2_Negotiate_Protocol_Response, Command=0, Flags=1)

class SMB2_Session_Setup_Request(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 Session Setup Request'
    OFFSET = 24 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 25), FlagsField('Flags', 0, -8, ['SMB2_SESSION_FLAG_BINDING']), FlagsField('SecurityMode', 0, -8, {1: 'Signing Required', 2: 'Signing Enabled'}), FlagsField('Capabilities', 0, -32, SMB2_CAPABILITIES), LEIntField('Channel', 0), XLEShortField('SecurityBufferOffset', None), LEShortField('SecurityLen', None), XLELongField('PreviousSessionId', 0), _NTLMPayloadField('Buffer', OFFSET, [PacketField('Security', None, GSSAPI_BLOB)])]

    def __getattr__(self, attr):
        if False:
            return 10
        if attr == 'SecurityBlob':
            return (super(SMB2_Session_Setup_Request, self).__getattr__('Buffer') or [(None, None)])[0][1]
        return super(SMB2_Session_Setup_Request, self).__getattr__(attr)

    def setfieldval(self, attr, val):
        if False:
            return 10
        if attr == 'SecurityBlob':
            return super(SMB2_Session_Setup_Request, self).setfieldval('Buffer', [('Security', val)])
        return super(SMB2_Session_Setup_Request, self).setfieldval(attr, val)

    def post_build(self, pkt, pay):
        if False:
            print('Hello World!')
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Security': 12}) + pay
bind_top_down(SMB2_Header, SMB2_Session_Setup_Request, Command=1)

class SMB2_Session_Setup_Response(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 Session Setup Response'
    OFFSET = 8 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 9), FlagsField('SessionFlags', 0, -16, {1: 'IS_GUEST', 2: 'IS_NULL', 4: 'ENCRYPT_DATE'}), XLEShortField('SecurityBufferOffset', None), LEShortField('SecurityLen', None), _NTLMPayloadField('Buffer', OFFSET, [PacketField('Security', None, GSSAPI_BLOB)])]

    def __getattr__(self, attr):
        if False:
            return 10
        if attr == 'SecurityBlob':
            return (super(SMB2_Session_Setup_Response, self).__getattr__('Buffer') or [(None, None)])[0][1]
        return super(SMB2_Session_Setup_Response, self).__getattr__(attr)

    def setfieldval(self, attr, val):
        if False:
            print('Hello World!')
        if attr == 'SecurityBlob':
            return super(SMB2_Session_Setup_Response, self).setfieldval('Buffer', [('Security', val)])
        return super(SMB2_Session_Setup_Response, self).setfieldval(attr, val)

    def post_build(self, pkt, pay):
        if False:
            for i in range(10):
                print('nop')
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Security': 4}) + pay
bind_top_down(SMB2_Header, SMB2_Session_Setup_Response, Command=1, Flags=1)

class SMB2_Session_Logoff_Request(_SMB2_Payload):
    name = 'SMB2 LOGOFF Request'
    fields_desc = [XLEShortField('StructureSize', 4), ShortField('reserved', 0)]
bind_top_down(SMB2_Header, SMB2_Session_Logoff_Request, Command=2)

class SMB2_Session_Logoff_Response(_SMB2_Payload):
    name = 'SMB2 LOGOFF Request'
    fields_desc = [XLEShortField('StructureSize', 4), ShortField('reserved', 0)]
bind_top_down(SMB2_Header, SMB2_Session_Logoff_Response, Command=2, Flags=1)

class SMB2_Tree_Connect_Request(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 TREE_CONNECT Request'
    OFFSET = 8 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 9), FlagsField('Flags', 0, -16, ['CLUSTER_RECONNECT', 'REDIRECT_TO_OWNER', 'EXTENSION_PRESENT']), XLEShortField('PathBufferOffset', None), LEShortField('PathLen', None), _NTLMPayloadField('Buffer', OFFSET, [StrFieldUtf16('Path', b'')])]

    def post_build(self, pkt, pay):
        if False:
            print('Hello World!')
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Path': 4}) + pay
bind_top_down(SMB2_Header, SMB2_Tree_Connect_Request, Command=3)
SMB2_ACCESS_FLAGS = {1: 'FILE_LIST_DIRECTORY', 2: 'FILE_ADD_FILE', 4: 'FILE_ADD_SUBDIRECTORY', 8: 'FILE_READ_EA', 16: 'FILE_WRITE_EA', 32: 'FILE_TRAVERSE', 64: 'FILE_DELETE_CHILD', 128: 'FILE_READ_ATTRIBUTES', 256: 'FILE_WRITE_ATTRIBUTES', 65536: 'DELETE', 131072: 'READ_CONTROL', 262144: 'WRITE_DAC', 524288: 'WRITE_OWNER', 1048576: 'SYNCHRONIZE', 16777216: 'ACCESS_SYSTEM_SECURITY', 33554432: 'MAXIMUM_ALLOWED', 268435456: 'GENERIC_ALL', 536870912: 'GENERIC_EXECUTE', 1073741824: 'GENERIC_WRITE', 2147483648: 'GENERIC_READ'}

class SMB2_Tree_Connect_Response(_SMB2_Payload):
    name = 'SMB2 TREE_CONNECT Response'
    fields_desc = [XLEShortField('StructureSize', 16), ByteEnumField('ShareType', 0, {1: 'DISK', 2: 'PIPE', 3: 'PRINT'}), ByteField('Reserved', 0), FlagsField('ShareFlags', 48, -32, {16: 'AUTO_CACHING', 32: 'VDO_CACHING', 48: 'NO_CACHING', 1: 'DFS', 2: 'DFS_ROOT', 256: 'RESTRICT_EXCLUSIVE_OPENS', 512: 'FORCE_SHARED_DELETE', 1024: 'ALLOW_NAMESPACE_CACHING', 2048: 'ACCESS_BASED_DIRECTORY_ENUM', 4096: 'FORCE_LEVELII_OPLOCK', 8192: 'ENABLE_HASH_V1', 16384: 'ENABLE_HASH_V2', 32768: 'ENCRYPT_DATA', 262144: 'IDENTITY_REMOTING', 1048576: 'COMPRESS_DATA'}), FlagsField('Capabilities', 0, -32, {8: 'DFS', 16: 'CONTINUOUS_AVAILABILITY', 32: 'SCALEOUT', 64: 'CLUSTER', 128: 'ASYMMETRIC', 256: 'REDIRECT_TO_OWNER'}), FlagsField('MaximalAccess', 0, -32, SMB2_ACCESS_FLAGS)]
bind_top_down(SMB2_Header, SMB2_Tree_Connect_Response, Command=3, Flags=1)

class SMB2_Tree_Disconnect_Request(_SMB2_Payload):
    name = 'SMB2 TREE_DISCONNECT Request'
    fields_desc = [XLEShortField('StructureSize', 4), XLEShortField('Reserved', 0)]
bind_top_down(SMB2_Header, SMB2_Tree_Disconnect_Request, Command=4)

class SMB2_Tree_Disconnect_Response(_SMB2_Payload):
    name = 'SMB2 TREE_DISCONNECT Response'
    fields_desc = [XLEShortField('StructureSize', 4), XLEShortField('Reserved', 0)]
bind_top_down(SMB2_Header, SMB2_Tree_Disconnect_Response, Command=4, Flags=1)

class SMB2_FILEID(Packet):
    fields_desc = [XLELongField('Persistent', 0), XLELongField('Volatile', 0)]

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Persistent + self.Volatile << 64

    def default_payload_class(self, payload):
        if False:
            return 10
        return conf.padding_layer

class SMB2_CREATE_DURABLE_HANDLE_RESPONSE(Packet):
    fields_desc = [XStrFixedLenField('Reserved', b'\x00' * 8, length=8)]

class SMB2_CREATE_QUERY_MAXIMAL_ACCESS_RESPONSE(Packet):
    fields_desc = [LEIntEnumField('QueryStatus', 0, STATUS_ERREF), FlagsField('MaximalAccess', 0, -32, SMB2_ACCESS_FLAGS)]

class SMB2_CREATE_QUERY_ON_DISK_ID(Packet):
    fields_desc = [LELongField('DiskFileId', 0), LELongField('VolumeId', 0), XStrFixedLenField('Reserved', b'', length=16)]

class SMB2_CREATE_RESPONSE_LEASE(Packet):
    fields_desc = [XStrFixedLenField('LeaseKey', b'', length=16), FlagsField('LeaseState', 7, -32, {1: 'SMB2_LEASE_READ_CACHING', 2: 'SMB2_LEASE_HANDLE_CACHING', 4: 'SMB2_LEASE_WRITE_CACHING'}), FlagsField('LeaseFlags', 0, -32, {2: 'SMB2_LEASE_FLAG_BREAK_IN_PROGRESS', 4: 'SMB2_LEASE_FLAG_PARENT_LEASE_KEY_SET'}), LELongField('LeaseDuration', 0)]

class SMB2_CREATE_RESPONSE_LEASE_V2(Packet):
    fields_desc = [SMB2_CREATE_RESPONSE_LEASE, XStrFixedLenField('ParentLeaseKey', b'', length=16), LEShortField('Epoch', 0), LEShortField('Reserved', 0)]

class SMB2_CREATE_DURABLE_HANDLE_RESPONSE_V2(Packet):
    fields_desc = [LEIntField('Timeout', 0), FlagsField('Flags', 0, -32, {2: 'SMB2_DHANDLE_FLAG_PERSISTENT'})]

class SMB2_CREATE_DURABLE_HANDLE_REQUEST(Packet):
    fields_desc = [XStrFixedLenField('DurableRequest', b'', length=16)]

class SMB2_CREATE_DURABLE_HANDLE_RECONNECT(Packet):
    fields_desc = [PacketField('Data', SMB2_FILEID(), SMB2_FILEID)]

class SMB2_CREATE_QUERY_MAXIMAL_ACCESS_REQUEST(Packet):
    fields_desc = [LELongField('Timestamp', 0)]

class SMB2_CREATE_ALLOCATION_SIZE(Packet):
    fields_desc = [LELongField('AllocationSize', 0)]

class SMB2_CREATE_TIMEWARP_TOKEN(Packet):
    fields_desc = [LELongField('Timestamp', 0)]

class SMB2_CREATE_REQUEST_LEASE(Packet):
    fields_desc = [SMB2_CREATE_RESPONSE_LEASE]

class SMB2_CREATE_REQUEST_LEASE_V2(Packet):
    fields_desc = [SMB2_CREATE_RESPONSE_LEASE_V2]

class SMB2_CREATE_DURABLE_HANDLE_REQUEST_V2(Packet):
    fields_desc = [SMB2_CREATE_DURABLE_HANDLE_RESPONSE_V2, XStrFixedLenField('Reserved', b'', length=8), UUIDField('CreateGuid', 0, uuid_fmt=UUIDField.FORMAT_LE)]

class SMB2_CREATE_DURABLE_HANDLE_RECONNECT_V2(Packet):
    fields_desc = [PacketField('FileId', SMB2_FILEID(), SMB2_FILEID), UUIDField('CreateGuid', 0, uuid_fmt=UUIDField.FORMAT_LE), FlagsField('Flags', 0, -32, {2: 'SMB2_DHANDLE_FLAG_PERSISTENT'})]

class SMB2_CREATE_APP_INSTANCE_ID(Packet):
    fields_desc = [XLEShortField('StructureSize', 20), LEShortField('Reserved', 0), XStrFixedLenField('AppInstanceId', b'', length=16)]

class SMB2_CREATE_APP_INSTANCE_VERSION(Packet):
    fields_desc = [XLEShortField('StructureSize', 24), LEShortField('Reserved', 0), LEIntField('Padding', 0), LELongField('AppInstanceVersionHigh', 0), LELongField('AppInstanceVersionLow', 0)]

class SMB2_Create_Context(_NTLMPayloadPacket):
    name = 'SMB2 CREATE CONTEXT'
    OFFSET = 14
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [LEIntField('Next', None), XLEShortField('NameBufferOffset', None), LEShortField('NameLen', None), ShortField('Reserved', 0), XLEShortField('DataBufferOffset', None), LEShortField('DataLen', None), _NTLMPayloadField('Buffer', OFFSET, [StrLenField('Name', b'', length_from=lambda pkt: pkt.NameLen), PacketLenField('Data', None, conf.raw_layer, length_from=lambda pkt: pkt.DataLen)]), StrLenField('pad', b'', length_from=lambda x: x.Next - max(x.DataBufferOffset + x.DataLen, x.NameBufferOffset + x.NameLen) if x.Next else 0)]

    def post_dissect(self, s):
        if False:
            print('Hello World!')
        if not self.DataLen:
            return s
        try:
            if isinstance(self.parent, SMB2_Create_Request):
                data_cls = {b'DHnQ': SMB2_CREATE_DURABLE_HANDLE_REQUEST, b'DHnC': SMB2_CREATE_DURABLE_HANDLE_RECONNECT, b'AISi': SMB2_CREATE_ALLOCATION_SIZE, b'MxAc': SMB2_CREATE_QUERY_MAXIMAL_ACCESS_REQUEST, b'TWrp': SMB2_CREATE_TIMEWARP_TOKEN, b'QFid': SMB2_CREATE_QUERY_ON_DISK_ID, b'RqLs': SMB2_CREATE_REQUEST_LEASE, b'DH2Q': SMB2_CREATE_DURABLE_HANDLE_REQUEST_V2, b'DH2C': SMB2_CREATE_DURABLE_HANDLE_RECONNECT_V2, b'E\xbc\xa6j\xef\xa7\xf7J\x90\x08\xfaF.\x14Mt': SMB2_CREATE_APP_INSTANCE_ID, b'\xb9\x82\xd0\xb7;V\x07O\xa0{RJ\x81\x16\xa0\x10': SMB2_CREATE_APP_INSTANCE_VERSION}[self.Name]
                if self.Name == b'RqLs' and self.DataLen > 32:
                    data_cls = SMB2_CREATE_REQUEST_LEASE_V2
            elif isinstance(self.parent, SMB2_Create_Response):
                data_cls = {b'DHnQ': SMB2_CREATE_DURABLE_HANDLE_RESPONSE, b'MxAc': SMB2_CREATE_QUERY_MAXIMAL_ACCESS_RESPONSE, b'QFid': SMB2_CREATE_QUERY_ON_DISK_ID, b'RqLs': SMB2_CREATE_RESPONSE_LEASE, b'DH2Q': SMB2_CREATE_DURABLE_HANDLE_RESPONSE_V2}[self.Name]
                if self.Name == b'RqLs' and self.DataLen > 32:
                    data_cls = SMB2_CREATE_RESPONSE_LEASE_V2
            else:
                return s
        except KeyError:
            return s
        self.Data = data_cls(self.Data.load)
        return s

    def default_payload_class(self, _):
        if False:
            return 10
        return conf.padding_layer

    def post_build(self, pkt, pay):
        if False:
            print('Hello World!')
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Name': 4, 'Data': 10}) + pay
SMB2_OPLOCK_LEVELS = {0: 'SMB2_OPLOCK_LEVEL_NONE', 1: 'SMB2_OPLOCK_LEVEL_II', 8: 'SMB2_OPLOCK_LEVEL_EXCLUSIVE', 9: 'SMB2_OPLOCK_LEVEL_BATCH', 255: 'SMB2_OPLOCK_LEVEL_LEASE'}

class SMB2_Create_Request(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 CREATE Request'
    OFFSET = 56 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 57), ByteField('ShareType', 0), ByteEnumField('RequestedOplockLevel', 0, SMB2_OPLOCK_LEVELS), LEIntEnumField('ImpersonationLevel', 0, {0: 'Anonymous', 1: 'Identification', 2: 'Impersonation', 3: 'Delegate'}), LELongField('SmbCreateFlags', 0), LELongField('Reserved', 0), FlagsField('DesiredAccess', 0, -32, SMB2_ACCESS_FLAGS), FlagsField('FileAttributes', 128, -32, FileAttributes), FlagsField('ShareAccess', 0, -32, {1: 'FILE_SHARE_READ', 2: 'FILE_SHARE_WRITE', 4: 'FILE_SHARE_DELETE'}), LEIntEnumField('CreateDisposition', 1, {0: 'FILE_SUPERSEDE', 1: 'FILE_OPEN', 2: 'FILE_CREATE', 3: 'FILE_OPEN_IF', 4: 'FILE_OVERWRITE', 5: 'FILE_OVERWRITE_IF'}), FlagsField('CreateOptions', 0, -32, {1: 'FILE_DIRECTORY_FILE', 2: 'FILE_WRITE_THROUGH', 4: 'FILE_SEQUENTIAL_ONLY', 8: 'FILE_NO_INTERMEDIATE_BUFFERING', 16: 'FILE_SYNCHRONOUS_IO_ALERT', 32: 'FILE_SYNCHRONOUS_IO_NONALERT', 64: 'FILE_NON_DIRECTORY_FILE', 256: 'FILE_COMPLETE_IF_OPLOCKED', 512: 'FILE_RANDOM_ACCESS', 4096: 'FILE_DELETE_ON_CLOSE', 8192: 'FILE_OPEN_BY_FILE_ID', 16384: 'FILE_OPEN_FOR_BACKUP_INTENT', 32768: 'FILE_NO_COMPRESSION', 1024: 'FILE_OPEN_REMOTE_INSTANCE', 65536: 'FILE_OPEN_REQUIRING_OPLOCK', 131072: 'FILE_DISALLOW_EXCLUSIVE', 1048576: 'FILE_RESERVE_OPFILTER', 2097152: 'FILE_OPEN_REPARSE_POINT', 4194304: 'FILE_OPEN_NO_RECALL', 8388608: 'FILE_OPEN_FOR_FREE_SPACE_QUERY'}), XLEShortField('NameBufferOffset', None), LEShortField('NameLen', None), XLEIntField('CreateContextsBufferOffset', None), LEIntField('CreateContextsLen', None), _NTLMPayloadField('Buffer', OFFSET, [StrFieldUtf16('Name', b''), _NextPacketListField('CreateContexts', [], SMB2_Create_Context, length_from=lambda pkt: pkt.CreateContextsLen)])]

    def post_build(self, pkt, pay):
        if False:
            return 10
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Name': 44, 'CreateContexts': 48}) + pay
bind_top_down(SMB2_Header, SMB2_Create_Request, Command=5)

class SMB2_Create_Response(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 CREATE Response'
    OFFSET = 88 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 89), ByteEnumField('OplockLevel', 0, SMB2_OPLOCK_LEVELS), FlagsField('Flags', 0, -8, {1: 'SMB2_CREATE_FLAG_REPARSEPOINT'}), LEIntEnumField('CreateAction', 1, {0: 'FILE_SUPERSEDED', 1: 'FILE_OPENED', 2: 'FILE_CREATED', 3: 'FILE_OVERWRITEN'}), FileNetworkOpenInformation, PacketField('FileId', SMB2_FILEID(), SMB2_FILEID), XLEIntField('CreateContextsBufferOffset', None), LEIntField('CreateContextsLen', None), _NTLMPayloadField('Buffer', OFFSET, [_NextPacketListField('CreateContexts', [], SMB2_Create_Context, length_from=lambda pkt: pkt.CreateContextsLen)])]

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        return _SMB2_post_build(self, pkt, self.OFFSET, {'CreateContexts': 80}) + pay
bind_top_down(SMB2_Header, SMB2_Create_Response, Command=5, Flags=1)

class SMB2_Close_Request(_SMB2_Payload):
    name = 'SMB2 CLOSE Request'
    fields_desc = [XLEShortField('StructureSize', 24), FlagsField('Flags', 0, -16, ['SMB2_CLOSE_FLAG_POSTQUERY_ATTRIB']), LEIntField('Reserved', 0), PacketField('FileId', SMB2_FILEID(), SMB2_FILEID)]
bind_top_down(SMB2_Header, SMB2_Close_Request, Command=6)

class SMB2_Close_Response(_SMB2_Payload):
    name = 'SMB2 CLOSE Response'
    FileAttributes = 0
    CreationTime = 0
    LastAccessTime = 0
    LastWriteTime = 0
    ChangeTime = 0
    fields_desc = [XLEShortField('StructureSize', 60), FlagsField('Flags', 0, -16, ['SMB2_CLOSE_FLAG_POSTQUERY_ATTRIB']), LEIntField('Reserved', 0)] + FileNetworkOpenInformation.fields_desc[:7]
bind_top_down(SMB2_Header, SMB2_Close_Response, Command=6, Flags=1)

class SMB2_Read_Request(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 READ Request'
    OFFSET = 48 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 49), ByteField('Padding', 0), FlagsField('Flags', 0, -8, {1: 'SMB2_READFLAG_READ_UNBUFFERED', 2: 'SMB2_READFLAG_REQUEST_COMPRESSED'}), LEIntField('Length', 0), LELongField('Offset', 0), PacketField('FileId', SMB2_FILEID(), SMB2_FILEID), LEIntField('MinimumCount', 0), LEIntEnumField('Channel', 0, {0: 'SMB2_CHANNEL_NONE', 1: 'SMB2_CHANNEL_RDMA_V1', 2: 'SMB2_CHANNEL_RDMA_V1_INVALIDATE', 3: 'SMB2_CHANNEL_RDMA_TRANSFORM'}), LEIntField('RemainingBytes', 0), LEShortField('ReadChannelInfoBufferOffset', None), LEShortField('ReadChannelInfoLen', None), _NTLMPayloadField('Buffer', OFFSET, [StrLenField('ReadChannelInfo', b'', length_from=lambda pkt: pkt.ReadChannelInfoLen)])]

    def post_build(self, pkt, pay):
        if False:
            for i in range(10):
                print('nop')
        return _SMB2_post_build(self, pkt, self.OFFSET, {'ReadChannelInfo': 44}) + pay
bind_top_down(SMB2_Header, SMB2_Read_Request, Command=8)

class SMB2_Read_Response(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 READ Response'
    OFFSET = 16 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 17), LEShortField('DataBufferOffset', None), LEIntField('DataLen', None), LEIntField('DataRemaining', 0), FlagsField('Flags', 0, -32, {1: 'SMB2_READFLAG_RESPONSE_RDMA_TRANSFORM'}), _NTLMPayloadField('Buffer', OFFSET, [StrLenField('Data', b'', length_from=lambda pkt: pkt.DataLen)])]

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Data': 2}) + pay
bind_top_down(SMB2_Header, SMB2_Read_Response, Command=8, Flags=1)

class SMB2_Write_Request(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 WRITE Request'
    OFFSET = 48 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 49), LEShortField('DataBufferOffset', None), LEIntField('DataLen', None), LELongField('Offset', 0), PacketField('FileId', SMB2_FILEID(), SMB2_FILEID), LEIntEnumField('Channel', 0, {0: 'SMB2_CHANNEL_NONE', 1: 'SMB2_CHANNEL_RDMA_V1', 2: 'SMB2_CHANNEL_RDMA_V1_INVALIDATE', 3: 'SMB2_CHANNEL_RDMA_TRANSFORM'}), LEIntField('RemainingBytes', 0), LEShortField('WriteChannelInfoBufferOffset', None), LEShortField('WriteChannelInfoLen', None), FlagsField('Flags', 0, -32, {1: 'SMB2_WRITEFLAG_WRITE_THROUGH', 2: 'SMB2_WRITEFLAG_WRITE_UNBUFFERED'}), _NTLMPayloadField('Buffer', OFFSET, [StrLenField('Data', b'', length_from=lambda pkt: pkt.DataLen), StrLenField('WriteChannelInfo', b'', length_from=lambda pkt: pkt.WriteChannelInfoLen)])]

    def post_build(self, pkt, pay):
        if False:
            for i in range(10):
                print('nop')
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Data': 2, 'WriteChannelInfo': 40}) + pay
bind_top_down(SMB2_Header, SMB2_Write_Request, Command=9)

class SMB2_Write_Response(_SMB2_Payload):
    name = 'SMB2 WRITE Response'
    fields_desc = [XLEShortField('StructureSize', 17), LEShortField('Reserved', 0), LEIntField('Count', 0), LEIntField('Remaining', 0), LEShortField('WriteChannelInfoBufferOffset', 0), LEShortField('WriteChannelInfoLen', 0)]
bind_top_down(SMB2_Header, SMB2_Write_Response, Command=9, Flags=1)

class SMB2_Cancel_Request(_SMB2_Payload):
    name = 'SMB2 CANCEL Request'
    fields_desc = [XLEShortField('StructureSize', 4), LEShortField('Reserved', 0)]
bind_top_down(SMB2_Header, SMB2_Cancel_Request, Command=9)

class SMB2_IOCTL_Validate_Negotiate_Info_Request(Packet):
    name = 'SMB2 IOCTL Validate Negotiate Info'
    fields_desc = SMB2_Negotiate_Protocol_Request.fields_desc[4:6] + SMB2_Negotiate_Protocol_Request.fields_desc[1:3][::-1] + [SMB2_Negotiate_Protocol_Request.fields_desc[9]]

class _SMB2_IOCTL_Request_PacketLenField(PacketLenField):

    def m2i(self, pkt, m):
        if False:
            for i in range(10):
                print('nop')
        if pkt.CtlCode == 1311236:
            return SMB2_IOCTL_Validate_Negotiate_Info_Request(m)
        return conf.raw_layer(m)

class SMB2_IOCTL_Request(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 IOCTL Request'
    OFFSET = 56 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    deprecated_fields = {'IntputCount': ('InputLen', 'alias'), 'OutputCount': ('OutputLen', 'alias')}
    fields_desc = [XLEShortField('StructureSize', 57), LEShortField('Reserved', 0), LEIntEnumField('CtlCode', 0, {393620: 'FSCTL_DFS_GET_REFERRALS', 1130508: 'FSCTL_PIPE_PEEK', 1114136: 'FSCTL_PIPE_WAIT', 1163287: 'FSCTL_PIPE_TRANSCEIVE', 1327346: 'FSCTL_SRV_COPYCHUNK', 1327204: 'FSCTL_SRV_ENUMERATE_SNAPSHOTS', 1310840: 'FSCTL_SRV_REQUEST_RESUME_KEY', 1327547: 'FSCTL_SRV_READ_HASH', 1343730: 'FSCTL_SRV_COPYCHUNK_WRITE', 1311188: 'FSCTL_LMR_REQUEST_RESILIENCY', 1311228: 'FSCTL_QUERY_NETWORK_INTERFACE_INFO', 589988: 'FSCTL_SET_REPARSE_POINT', 393648: 'FSCTL_DFS_GET_REFERRALS_EX', 623112: 'FSCTL_FILE_LEVEL_TRIM', 1311236: 'FSCTL_VALIDATE_NEGOTIATE_INFO'}), PacketField('FileId', SMB2_FILEID(), SMB2_FILEID), LEIntField('InputBufferOffset', None), LEIntField('InputLen', None), LEIntField('MaxInputResponse', 0), LEIntField('OutputBufferOffset', None), LEIntField('OutputLen', None), LEIntField('MaxOutputResponse', 0), FlagsField('Flags', 0, -32, {1: 'SMB2_0_IOCTL_IS_FSCTL'}), LEIntField('Reserved2', 0), _NTLMPayloadField('Buffer', OFFSET, [_SMB2_IOCTL_Request_PacketLenField('Input', None, conf.raw_layer, length_from=lambda pkt: pkt.InputLen), _SMB2_IOCTL_Request_PacketLenField('Output', None, conf.raw_layer, length_from=lambda pkt: pkt.OutputLen)])]

    def post_build(self, pkt, pay):
        if False:
            i = 10
            return i + 15
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Input': 24, 'Output': 36}) + pay
bind_top_down(SMB2_Header, SMB2_IOCTL_Request, Command=11)

class SMB2_IOCTL_Validate_Negotiate_Info_Response(Packet):
    name = 'SMB2 IOCTL Validate Negotiate Info'
    fields_desc = SMB2_Negotiate_Protocol_Response.fields_desc[4:6][::-1] + SMB2_Negotiate_Protocol_Response.fields_desc[1:3]

class _SMB2_IOCTL_Response_PacketLenField(PacketLenField):

    def m2i(self, pkt, m):
        if False:
            print('Hello World!')
        if pkt.CtlCode == 1311236:
            return SMB2_IOCTL_Validate_Negotiate_Info_Response(m)
        return conf.raw_layer(m)

class SMB2_IOCTL_Response(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 IOCTL Response'
    OFFSET = 48 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    StructureSize = 49
    fields_desc = SMB2_IOCTL_Request.fields_desc[:6] + SMB2_IOCTL_Request.fields_desc[7:9] + SMB2_IOCTL_Request.fields_desc[10:12] + [_NTLMPayloadField('Buffer', OFFSET, [_SMB2_IOCTL_Response_PacketLenField('Input', None, conf.raw_layer, length_from=lambda pkt: pkt.InputLen), _SMB2_IOCTL_Response_PacketLenField('Output', None, conf.raw_layer, length_from=lambda pkt: pkt.OutputLen)])]

    def post_build(self, pkt, pay):
        if False:
            print('Hello World!')
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Input': 24, 'Output': 32}) + pay
bind_top_down(SMB2_Header, SMB2_IOCTL_Response, Command=11, Flags=1)

class SMB2_Query_Directory_Request(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 QUERY DIRECTORY Request'
    OFFSET = 32 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 33), ByteEnumField('FileInformationClass', 1, FileInformationClasses), FlagsField('Flags', 0, -8, {1: 'SMB2_RESTART_SCANS', 2: 'SMB2_RETURN_SINGLE_ENTRY', 4: 'SMB2_INDEX_SPECIFIED', 16: 'SMB2_REOPEN'}), LEIntField('FileIndex', 0), PacketField('FileId', SMB2_FILEID(), SMB2_FILEID), LEShortField('FileNameBufferOffset', None), LEShortField('FileNameLen', None), LEIntField('OutputBufferLength', 2048), _NTLMPayloadField('Buffer', OFFSET, [StrFieldUtf16('FileName', b'')])]

    def post_build(self, pkt, pay):
        if False:
            print('Hello World!')
        return _SMB2_post_build(self, pkt, self.OFFSET, {'FileName': 24}) + pay
bind_top_down(SMB2_Header, SMB2_Query_Directory_Request, Command=14)

class SMB2_Query_Directory_Response(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 QUERY DIRECTORY Response'
    OFFSET = 8 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 9), LEShortField('OutputBufferOffset', None), LEIntField('OutputLen', None), _NTLMPayloadField('Buffer', OFFSET, [StrFixedLenField('Output', b'', length_from=lambda pkt: pkt.OutputLen)])]

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Output': 2}) + pay
bind_top_down(SMB2_Header, SMB2_Query_Directory_Response, Command=14, Flags=1)

class SMB2_Change_Notify_Request(_SMB2_Payload):
    name = 'SMB2 CHANGE NOTIFY Request'
    fields_desc = [XLEShortField('StructureSize', 32), FlagsField('Flags', 0, -16, {1: 'SMB2_WATCH_TREE'}), LEIntField('OutputBufferLength', 2048), PacketField('FileId', SMB2_FILEID(), SMB2_FILEID), FlagsField('CompletionFilter', 0, -32, {1: 'FILE_NOTIFY_CHANGE_FILE_NAME', 2: 'FILE_NOTIFY_CHANGE_DIR_NAME', 4: 'FILE_NOTIFY_CHANGE_ATTRIBUTES', 8: 'FILE_NOTIFY_CHANGE_SIZE', 16: 'FILE_NOTIFY_CHANGE_LAST_WRITE', 32: 'FILE_NOTIFY_CHANGE_LAST_ACCESS', 64: 'FILE_NOTIFY_CHANGE_CREATION', 128: 'FILE_NOTIFY_CHANGE_EA', 256: 'FILE_NOTIFY_CHANGE_SECURITY', 512: 'FILE_NOTIFY_CHANGE_STREAM_NAME', 1024: 'FILE_NOTIFY_CHANGE_STREAM_SIZE', 2048: 'FILE_NOTIFY_CHANGE_STREAM_WRITE'}), LEIntField('Reserved', 0)]
bind_top_down(SMB2_Header, SMB2_Change_Notify_Request, Command=15)

class SMB2_Change_Notify_Response(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 CHANGE NOTIFY Response'
    OFFSET = 8 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 9), LEShortField('OutputBufferOffset', None), LEIntField('OutputLen', None), _NTLMPayloadField('Buffer', OFFSET, [StrFixedLenField('Output', b'', length_from=lambda pkt: pkt.OutputLen)])]

    def post_build(self, pkt, pay):
        if False:
            return 10
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Output': 2}) + pay
bind_top_down(SMB2_Header, SMB2_Change_Notify_Response, Command=15, Flags=1)

class FILE_GET_QUOTA_INFORMATION(Packet):
    fields_desc = [IntField('NextEntryOffset', 0), FieldLenField('SidLength', None, length_of='Sid'), StrLenField('Sid', b'', length_from=lambda x: x.SidLength), StrLenField('pad', b'', length_from=lambda x: x.NextEntryOffset - x.SidLength if x.NextEntryOffset else 0)]

class SMB2_Query_Quota_Info(Packet):
    fields_desc = [ByteField('ReturnSingle', 0), ByteField('ReturnBoolean', 0), ShortField('Reserved', 0), LEIntField('SidListLength', 0), LEIntField('StartSidLength', 0), LEIntField('StartSidOffset', 0), StrLenField('pad', b'', length_from=lambda x: x.StartSidOffset), MultipleTypeField([(PacketListField('SidBuffer', [], FILE_GET_QUOTA_INFORMATION, length_from=lambda x: x.SidListLength), lambda x: x.SidListLength), (StrLenField('SidBuffer', b'', length_from=lambda x: x.StartSidLength), lambda x: x.StartSidLength)], StrFixedLenField('SidBuffer', b'', length=0))]

class SMB2_Query_Info_Request(_SMB2_Payload, _NTLMPayloadPacket):
    name = 'SMB2 QUERY INFO Request'
    OFFSET = 40 + 64
    _NTLM_PAYLOAD_FIELD_NAME = 'Buffer'
    fields_desc = [XLEShortField('StructureSize', 41), ByteEnumField('InfoType', 0, {1: 'SMB2_0_INFO_FILE', 2: 'SMB2_0_INFO_FILESYSTEM', 3: 'SMB2_0_INFO_SECURITY', 4: 'SMB2_0_INFO_QUOTA'}), ByteEnumField('FileInfoClass', 0, FileInformationClasses), LEIntField('OutputBufferLength', 0), XLEIntField('InputBufferOffset', None), LEIntField('InputLen', None), FlagsField('AdditionalInformation', 0, -32, {1: 'OWNER_SECURITY_INFORMATION', 2: 'GROUP_SECURITY_INFORMATION', 4: 'DACL_SECURITY_INFORMATION', 8: 'SACL_SECURITY_INFORMATION', 16: 'LABEL_SECURITY_INFORMATION', 32: 'ATTRIBUTE_SECURITY_INFORMATION', 64: 'SCOPE_SECURITY_INFORMATION', 65536: 'BACKUP_SECURITY_INFORMATION'}), FlagsField('Flags', 0, -32, {1: 'SL_RESTART_SCAN', 2: 'SL_RETURN_SINGLE_ENTRY', 4: 'SL_INDEX_SPECIFIED'}), PacketField('FileId', SMB2_FILEID(), SMB2_FILEID), _NTLMPayloadField('Buffer', OFFSET, [PacketListField('Input', None, SMB2_Query_Quota_Info, length_from=lambda pkt: pkt.InputLen)])]

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Input': 4}) + pay
bind_top_down(SMB2_Header, SMB2_Query_Info_Request, Command=16)

class SMB2_Query_Info_Response(_SMB2_Payload):
    name = 'SMB2 QUERY INFO Response'
    OFFSET = 8 + 64
    fields_desc = [XLEShortField('StructureSize', 9), LEShortField('OutputBufferOffset', None), LEIntField('OutputLen', None), _NTLMPayloadField('Buffer', OFFSET, [StrFixedLenField('Output', b'', length_from=lambda pkt: pkt.OutputLen)])]

    def post_build(self, pkt, pay):
        if False:
            return 10
        return _SMB2_post_build(self, pkt, self.OFFSET, {'Output': 2}) + pay
bind_top_down(SMB2_Header, SMB2_Query_Info_Response, Command=16, Flags=1)

class SMB2_Compression_Transform_Header(Packet):
    name = 'SMB2 Compression Transform Header'
    fields_desc = [StrFixedLenField('Start', b'\xfcSMB', 4), LEIntField('OriginalCompressedSegmentSize', 0), LEShortEnumField('CompressionAlgorithm', 0, SMB2_COMPRESSION_ALGORITHMS), ShortEnumField('Flags', 0, {0: 'SMB2_COMPRESSION_FLAG_NONE', 1: 'SMB2_COMPRESSION_FLAG_CHAINED'}), XLEIntField('Offset_or_Length', 0)]