"""
SMB (Server Message Block), also known as CIFS.

Specs:
- [MS-CIFS] (base)
- [MS-SMB] (extension of CIFS - SMB v1)
"""
import struct
from scapy.config import conf
from scapy.packet import Packet, bind_layers, bind_top_down
from scapy.fields import ByteEnumField, ByteField, FieldLenField, FlagsField, LEFieldLenField, LEIntEnumField, LEIntField, LEShortField, MultipleTypeField, PacketLenField, PacketListField, ReversePadField, ScalingField, ShortField, StrFixedLenField, StrNullField, StrNullFieldUtf16, UTCTimeField, UUIDField, XStrLenField
from scapy.layers.netbios import NBTSession
from scapy.layers.gssapi import GSSAPI_BLOB
from scapy.layers.smb2 import STATUS_ERREF, SMB2_Header
SMB_COM = {0: 'SMB_COM_CREATE_DIRECTORY', 1: 'SMB_COM_DELETE_DIRECTORY', 2: 'SMB_COM_OPEN', 3: 'SMB_COM_CREATE', 4: 'SMB_COM_CLOSE', 5: 'SMB_COM_FLUSH', 6: 'SMB_COM_DELETE', 7: 'SMB_COM_RENAME', 8: 'SMB_COM_QUERY_INFORMATION', 9: 'SMB_COM_SET_INFORMATION', 10: 'SMB_COM_READ', 11: 'SMB_COM_WRITE', 12: 'SMB_COM_LOCK_BYTE_RANGE', 13: 'SMB_COM_UNLOCK_BYTE_RANGE', 14: 'SMB_COM_CREATE_TEMPORARY', 15: 'SMB_COM_CREATE_NEW', 16: 'SMB_COM_CHECK_DIRECTORY', 17: 'SMB_COM_PROCESS_EXIT', 18: 'SMB_COM_SEEK', 19: 'SMB_COM_LOCK_AND_READ', 20: 'SMB_COM_WRITE_AND_UNLOCK', 26: 'SMB_COM_READ_RAW', 27: 'SMB_COM_READ_MPX', 28: 'SMB_COM_READ_MPX_SECONDARY', 29: 'SMB_COM_WRITE_RAW', 30: 'SMB_COM_WRITE_MPX', 31: 'SMB_COM_WRITE_MPX_SECONDARY', 32: 'SMB_COM_WRITE_COMPLETE', 33: 'SMB_COM_QUERY_SERVER', 34: 'SMB_COM_SET_INFORMATION2', 35: 'SMB_COM_QUERY_INFORMATION2', 36: 'SMB_COM_LOCKING_ANDX', 37: 'SMB_COM_TRANSACTION', 38: 'SMB_COM_TRANSACTION_SECONDARY', 39: 'SMB_COM_IOCTL', 40: 'SMB_COM_IOCTL_SECONDARY', 41: 'SMB_COM_COPY', 42: 'SMB_COM_MOVE', 43: 'SMB_COM_ECHO', 44: 'SMB_COM_WRITE_AND_CLOSE', 45: 'SMB_COM_OPEN_ANDX', 46: 'SMB_COM_READ_ANDX', 47: 'SMB_COM_WRITE_ANDX', 48: 'SMB_COM_NEW_FILE_SIZE', 49: 'SMB_COM_CLOSE_AND_TREE_DISC', 50: 'SMB_COM_TRANSACTION2', 51: 'SMB_COM_TRANSACTION2_SECONDARY', 52: 'SMB_COM_FIND_CLOSE2', 53: 'SMB_COM_FIND_NOTIFY_CLOSE', 112: 'SMB_COM_TREE_CONNECT', 113: 'SMB_COM_TREE_DISCONNECT', 114: 'SMB_COM_NEGOTIATE', 115: 'SMB_COM_SESSION_SETUP_ANDX', 116: 'SMB_COM_LOGOFF_ANDX', 117: 'SMB_COM_TREE_CONNECT_ANDX', 126: 'SMB_COM_SECURITY_PACKAGE_ANDX', 128: 'SMB_COM_QUERY_INFORMATION_DISK', 129: 'SMB_COM_SEARCH', 130: 'SMB_COM_FIND', 131: 'SMB_COM_FIND_UNIQUE', 132: 'SMB_COM_FIND_CLOSE', 160: 'SMB_COM_NT_TRANSACT', 161: 'SMB_COM_NT_TRANSACT_SECONDARY', 162: 'SMB_COM_NT_CREATE_ANDX', 164: 'SMB_COM_NT_CANCEL', 165: 'SMB_COM_NT_RENAME', 192: 'SMB_COM_OPEN_PRINT_FILE', 193: 'SMB_COM_WRITE_PRINT_FILE', 194: 'SMB_COM_CLOSE_PRINT_FILE', 195: 'SMB_COM_GET_PRINT_QUEUE', 216: 'SMB_COM_READ_BULK', 217: 'SMB_COM_WRITE_BULK', 218: 'SMB_COM_WRITE_BULK_DATA', 254: 'SMB_COM_INVALID', 255: 'SMB_COM_NO_ANDX_COMMAND'}

class SMB_Header(Packet):
    name = 'SMB 1 Protocol Request Header'
    fields_desc = [StrFixedLenField('Start', b'\xffSMB', 4), ByteEnumField('Command', 114, SMB_COM), LEIntEnumField('Status', 0, STATUS_ERREF), FlagsField('Flags', 24, 8, ['LOCK_AND_READ_OK', 'BUF_AVAIL', 'res', 'CASE_INSENSITIVE', 'CANONICALIZED_PATHS', 'OPLOCK', 'OPBATCH', 'REPLY']), FlagsField('Flags2', 0, -16, ['LONG_NAMES', 'EAS', 'SMB_SECURITY_SIGNATURE', 'COMPRESSED', 'SMB_SECURITY_SIGNATURE_REQUIRED', 'res', 'IS_LONG_NAME', 'res', 'res', 'res', 'REPARSE_PATH', 'EXTENDED_SECURITY', 'DFS', 'PAGING_IO', 'NT_STATUS', 'UNICODE']), LEShortField('PIDHigh', 0), StrFixedLenField('SecuritySignature', b'', length=8), LEShortField('Reserved', 0), LEShortField('TID', 0), LEShortField('PIDLow', 1), LEShortField('UID', 0), LEShortField('MID', 0)]

    def guess_payload_class(self, payload):
        if False:
            return 10
        if not payload:
            return super(SMB_Header, self).guess_payload_class(payload)
        WordCount = ord(payload[:1])
        if self.Command == 114:
            if self.Flags.REPLY:
                if self.Flags2.EXTENDED_SECURITY:
                    return SMBNegotiate_Response_Extended_Security
                else:
                    return SMBNegotiate_Response_Security
            else:
                return SMBNegotiate_Request
        elif self.Command == 115:
            if WordCount == 0:
                return SMBSession_Null
            if self.Flags.REPLY:
                if WordCount == 4:
                    return SMBSession_Setup_AndX_Response_Extended_Security
                elif WordCount == 3:
                    return SMBSession_Setup_AndX_Response
                if self.Flags2.EXTENDED_SECURITY:
                    return SMBSession_Setup_AndX_Response_Extended_Security
                else:
                    return SMBSession_Setup_AndX_Response
            else:
                if WordCount == 12:
                    return SMBSession_Setup_AndX_Request_Extended_Security
                elif WordCount == 13:
                    return SMBSession_Setup_AndX_Request
                if self.Flags2.EXTENDED_SECURITY:
                    return SMBSession_Setup_AndX_Request_Extended_Security
                else:
                    return SMBSession_Setup_AndX_Request
        elif self.Command == 37:
            return SMBNetlogon_Protocol_Response_Header
        return super(SMB_Header, self).guess_payload_class(payload)

    def answers(self, pkt):
        if False:
            return 10
        return SMB_Header in pkt

class SMB_Dialect(Packet):
    name = 'SMB Dialect'
    fields_desc = [ByteField('BufferFormat', 2), StrNullField('DialectString', 'NT LM 0.12')]

    def default_payload_class(self, payload):
        if False:
            for i in range(10):
                print('nop')
        return conf.padding_layer

class SMBNegotiate_Request(Packet):
    name = 'SMB Negotiate Request'
    fields_desc = [ByteField('WordCount', 0), LEFieldLenField('ByteCount', None, length_of='Dialects'), PacketListField('Dialects', [SMB_Dialect()], SMB_Dialect, length_from=lambda pkt: pkt.ByteCount)]
bind_layers(SMB_Header, SMBNegotiate_Request, Command=114)

def _SMBStrNullField(name, default):
    if False:
        i = 10
        return i + 15
    '\n    Returns a StrNullField that is either normal or UTF-16 depending\n    on the SMB headers.\n    '

    def _isUTF16(pkt):
        if False:
            i = 10
            return i + 15
        while not hasattr(pkt, 'Flags2') and pkt.underlayer:
            pkt = pkt.underlayer
        return hasattr(pkt, 'Flags2') and pkt.Flags2.UNICODE
    return MultipleTypeField([(StrNullFieldUtf16(name, default), _isUTF16)], StrNullField(name, default))

def _len(pkt, name):
    if False:
        print('Hello World!')
    '\n    Returns the length of a field, works with Unicode strings.\n    '
    (fld, v) = pkt.getfield_and_val(name)
    return len(fld.addfield(pkt, v, b''))

class _SMBNegotiate_Response(Packet):

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            print('Hello World!')
        if _pkt and len(_pkt) >= 2:
            wc = struct.unpack('<H', _pkt[:1])
            if wc == 1:
                return SMBNegotiate_Response_NoSecurity
            elif wc == 13:
                pass
            elif wc == 17:
                return cls
        return cls
_SMB_ServerCapabilities = ['RAW_MODE', 'MPX_MODE', 'UNICODE', 'LARGE_FILES', 'NT_SMBS', 'RPC_REMOTE_APIS', 'STATUS32', 'LEVEL_II_OPLOCKS', 'LOCK_AND_READ', 'NT_FIND', 'res', 'res', 'DFS', 'INFOLEVEL_PASSTHRU', 'LARGE_READX', 'LARGE_WRITEX', 'LWIO', 'res', 'res', 'res', 'res', 'res', 'res', 'UNIX', 'res', 'COMPRESSED_DATA', 'res', 'res', 'res', 'DYNAMIC_REAUTH', 'PERSISTENT_HANDLES', 'EXTENDED_SECURITY']

class SMBNegotiate_Response_NoSecurity(_SMBNegotiate_Response):
    name = 'SMB Negotiate No-Security Response (CIFS)'
    fields_desc = [ByteField('WordCount', 1), LEShortField('DialectIndex', 7), FlagsField('SecurityMode', 3, 8, ['USER_SECURITY', 'ENCRYPT_PASSWORDS', 'SECURITY_SIGNATURES_ENABLED', 'SECURITY_SIGNATURES_REQUIRED']), LEShortField('MaxMpxCount', 50), LEShortField('MaxNumberVC', 1), LEIntField('MaxBufferSize', 16144), LEIntField('MaxRawSize', 65536), LEIntField('SessionKey', 0), FlagsField('ServerCapabilities', 62457, -32, _SMB_ServerCapabilities), UTCTimeField('ServerTime', None, fmt='<Q', epoch=[1601, 1, 1, 0, 0, 0], custom_scaling=10000000.0), ScalingField('ServerTimeZone', 60, fmt='<h', unit='min-UTC'), FieldLenField('ChallengeLength', None, length_of='Challenge', fmt='<B'), LEFieldLenField('ByteCount', None, length_of='DomainName', adjust=lambda pkt, x: x + len(pkt.Challenge)), XStrLenField('Challenge', b'', length_from=lambda pkt: pkt.ChallengeLength), StrNullField('DomainName', 'WORKGROUP')]
bind_top_down(SMB_Header, SMBNegotiate_Response_NoSecurity, Command=114, Flags=128)

class SMBNegotiate_Response_Extended_Security(_SMBNegotiate_Response):
    name = 'SMB Negotiate Extended Security Response (SMB)'
    WordCount = 17
    fields_desc = SMBNegotiate_Response_NoSecurity.fields_desc[:12] + [LEFieldLenField('ByteCount', None, length_of='SecurityBlob', adjust=lambda _, x: x + 16), SMBNegotiate_Response_NoSecurity.fields_desc[13], UUIDField('GUID', None, uuid_fmt=UUIDField.FORMAT_LE), PacketLenField('SecurityBlob', None, GSSAPI_BLOB, length_from=lambda x: x.ByteCount - 16)]
bind_top_down(SMB_Header, SMBNegotiate_Response_Extended_Security, Command=114, Flags=128, Flags2=2048)

class SMBNegotiate_Response_Security(_SMBNegotiate_Response):
    name = 'SMB Negotiate Non-Extended Security Response (SMB)'
    WordCount = 17
    fields_desc = SMBNegotiate_Response_NoSecurity.fields_desc[:12] + [LEFieldLenField('ByteCount', None, length_of='DomainName', adjust=lambda pkt, x: x + 2 + _len(pkt, 'Challenge') + _len(pkt, 'ServerName')), XStrLenField('Challenge', b'', length_from=lambda pkt: pkt.ChallengeLength), _SMBStrNullField('DomainName', 'WORKGROUP'), _SMBStrNullField('ServerName', 'RMFF1')]
bind_top_down(SMB_Header, SMBNegotiate_Response_Security, Command=114, Flags=128)

class SMBSession_Setup_AndX_Request(Packet):
    name = 'Session Setup AndX Request (CIFS)'
    fields_desc = [ByteField('WordCount', 13), ByteEnumField('AndXCommand', 255, SMB_COM), ByteField('AndXReserved', 0), LEShortField('AndXOffset', None), LEShortField('MaxBufferSize', 16144), LEShortField('MaxMPXCount', 50), LEShortField('VCNumber', 0), LEIntField('SessionKey', 0), LEFieldLenField('OEMPasswordLength', None, length_of='OEMPassword'), LEFieldLenField('UnicodePasswordLength', None, length_of='UnicodePassword'), LEIntField('Reserved', 0), FlagsField('ServerCapabilities', 5, -32, _SMB_ServerCapabilities), LEShortField('ByteCount', None), XStrLenField('OEMPassword', 'Pass', length_from=lambda x: x.OEMPasswordLength), XStrLenField('UnicodePassword', 'Pass', length_from=lambda x: x.UnicodePasswordLength), ReversePadField(_SMBStrNullField('AccountName', 'GUEST'), 2, b'\x00'), _SMBStrNullField('PrimaryDomain', ''), _SMBStrNullField('NativeOS', 'Windows 4.0'), _SMBStrNullField('NativeLanMan', 'Windows 4.0')]

    def post_build(self, pkt, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.AndXOffset is None and self.AndXCommand != 255:
            pkt = pkt[:3] + struct.pack('<H', len(pkt) + 32) + pkt[5:]
        if self.ByteCount is None:
            pkt = pkt[:27] + struct.pack('<H', len(pkt) - 29) + pkt[29:]
        if self.payload and hasattr(self.payload, 'AndXOffset') and pay:
            pay = pay[:3] + struct.pack('<H', len(pkt) + len(pay) + 32) + pay[5:]
        return pkt + pay
bind_top_down(SMB_Header, SMBSession_Setup_AndX_Request, Command=115)

class SMBTree_Connect_AndX(Packet):
    name = 'Session Tree Connect AndX'
    WordCount = 4
    fields_desc = SMBSession_Setup_AndX_Request.fields_desc[:4] + [FlagsField('Flags', '', -16, ['DISCONNECT_TID', 'r2', 'EXTENDED_SIGNATURES', 'EXTENDED_RESPONSE']), FieldLenField('PasswordLength', None, length_of='Password', fmt='<H'), LEShortField('ByteCount', None), XStrLenField('Password', b'', length_from=lambda pkt: pkt.PasswordLength), ReversePadField(_SMBStrNullField('Path', '\\\\WIN2K\\IPC$'), 2), StrNullField('Service', '?????')]

    def post_build(self, pkt, pay):
        if False:
            i = 10
            return i + 15
        pkt += pay
        if self.ByteCount is None:
            pkt = pkt[:9] + struct.pack('<H', len(pkt) - 11) + pkt[11:]
        return pkt
bind_layers(SMB_Header, SMBTree_Connect_AndX, Command=117)
bind_layers(SMBSession_Setup_AndX_Request, SMBTree_Connect_AndX, AndXCommand=117)

class SMBSession_Setup_AndX_Request_Extended_Security(Packet):
    name = 'Session Setup AndX Extended Security Request (SMB)'
    WordCount = 12
    fields_desc = SMBSession_Setup_AndX_Request.fields_desc[:8] + [LEFieldLenField('SecurityBlobLength', None, length_of='SecurityBlob')] + SMBSession_Setup_AndX_Request.fields_desc[10:12] + [LEShortField('ByteCount', None), PacketLenField('SecurityBlob', None, GSSAPI_BLOB, length_from=lambda x: x.SecurityBlobLength), ReversePadField(_SMBStrNullField('NativeOS', 'Windows 4.0'), 2, b'\x00'), _SMBStrNullField('NativeLanMan', 'Windows 4.0')]

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        if self.ByteCount is None:
            pkt = pkt[:25] + struct.pack('<H', len(pkt) - 27) + pkt[27:]
        return pkt + pay
bind_top_down(SMB_Header, SMBSession_Setup_AndX_Request_Extended_Security, Command=115, Flags2=2048)

class SMBSession_Setup_AndX_Response(Packet):
    name = 'Session Setup AndX Response (CIFS)'
    fields_desc = [ByteField('WordCount', 3), ByteEnumField('AndXCommand', 255, SMB_COM), ByteField('AndXReserved', 0), LEShortField('AndXOffset', None), FlagsField('Action', 0, -16, {1: 'SMB_SETUP_GUEST', 2: 'SMB_SETUP_USE_LANMAN_KEY'}), LEShortField('ByteCount', 25), _SMBStrNullField('NativeOS', 'Windows 4.0'), _SMBStrNullField('NativeLanMan', 'Windows 4.0'), _SMBStrNullField('PrimaryDomain', ''), ByteField('WordCount2', 3), ByteEnumField('AndXCommand2', 255, SMB_COM), ByteField('Reserved3', 0), LEShortField('AndXOffset2', 80), LEShortField('OptionalSupport', 1), LEShortField('ByteCount2', 5), StrNullField('Service', 'IPC'), StrNullField('NativeFileSystem', '')]

    def post_build(self, pkt, pay):
        if False:
            return 10
        if self.AndXOffset is None:
            pkt = pkt[:3] + struct.pack('<H', len(pkt) + 32) + pkt[5:]
        return pkt + pay
bind_top_down(SMB_Header, SMBSession_Setup_AndX_Response, Command=115, Flags=128)

class SMBSession_Setup_AndX_Response_Extended_Security(SMBSession_Setup_AndX_Response):
    name = 'Session Setup AndX Extended Security Response (SMB)'
    WordCount = 4
    fields_desc = SMBSession_Setup_AndX_Response.fields_desc[:5] + [SMBSession_Setup_AndX_Request_Extended_Security.fields_desc[8]] + SMBSession_Setup_AndX_Request_Extended_Security.fields_desc[11:]

    def post_build(self, pkt, pay):
        if False:
            print('Hello World!')
        if self.ByteCount is None:
            pkt = pkt[:9] + struct.pack('<H', len(pkt) - 11) + pkt[11:]
        return super(SMBSession_Setup_AndX_Response_Extended_Security, self).post_build(pkt, pay)
bind_top_down(SMB_Header, SMBSession_Setup_AndX_Response_Extended_Security, Command=115, Flags=128, Flags2=2048)

class SMBSession_Null(Packet):
    fields_desc = [ByteField('WordCount', 0), LEShortField('ByteCount', 0)]
bind_top_down(SMB_Header, SMBSession_Null, Command=115)

class SMBNetlogon_Protocol_Response_Header(Packet):
    name = 'SMBNetlogon Protocol Response Header'
    fields_desc = [ByteField('WordCount', 17), LEShortField('TotalParamCount', 0), LEShortField('TotalDataCount', 112), LEShortField('MaxParamCount', 0), LEShortField('MaxDataCount', 0), ByteField('MaxSetupCount', 0), ByteField('unused2', 0), LEShortField('Flags3', 0), ByteField('TimeOut1', 232), ByteField('TimeOut2', 3), LEShortField('unused3', 0), LEShortField('unused4', 0), LEShortField('ParamCount2', 0), LEShortField('ParamOffset', 0), LEShortField('DataCount', 112), LEShortField('DataOffset', 92), ByteField('SetupCount', 3), ByteField('unused5', 0)]
bind_top_down(SMB_Header, SMBNetlogon_Protocol_Response_Header, Command=37)

class SMBMailSlot(Packet):
    name = 'SMB Mail Slot Protocol'
    fields_desc = [LEShortField('opcode', 1), LEShortField('priority', 1), LEShortField('class_', 2), LEShortField('size', 135), StrNullField('name', '\\MAILSLOT\\NET\\GETDC660')]

class SMBNetlogon_Protocol_Response_Tail_SAM(Packet):
    name = 'SMB Netlogon Protocol Response Tail SAM'
    fields_desc = [ByteEnumField('Command', 23, {18: 'SAM logon request', 23: 'SAM Active directory Response'}), ByteField('unused', 0), ShortField('Data1', 0), ShortField('Data2', 64769), ShortField('Data3', 0), ShortField('Data4', 44254), ShortField('Data5', 4069), ShortField('Data6', 53514), ShortField('Data7', 14156), ShortField('Data8', 33762), ShortField('Data9', 32217), ShortField('Data10', 14870), ShortField('Data11', 29695), ByteField('Data12', 4), StrFixedLenField('Data13', 'rmff', 4), ByteField('Data14', 0), ShortField('Data16', 49176), ByteField('Data18', 10), StrFixedLenField('Data20', 'rmff-win2k', 10), ByteField('Data21', 192), ShortField('Data22', 6336), ShortField('Data23', 6154), StrFixedLenField('Data24', 'RMFF-WIN2K', 10), ShortField('Data25', 0), ByteField('Data26', 23), StrFixedLenField('Data27', 'Default-First-Site-Name', 23), ShortField('Data28', 192), ShortField('Data29', 15376), ShortField('Data30', 192), ShortField('Data31', 512), ShortField('Data32', 0), ShortField('Data33', 44052), ShortField('Data34', 100), ShortField('Data35', 0), ShortField('Data36', 0), ShortField('Data37', 0), ShortField('Data38', 0), ShortField('Data39', 3328), ShortField('Data40', 0), ShortField('Data41', 65535)]

class SMBNetlogon_Protocol_Response_Tail_LM20(Packet):
    name = 'SMB Netlogon Protocol Response Tail LM20'
    fields_desc = [ByteEnumField('Command', 6, {6: 'LM 2.0 Response to logon request'}), ByteField('unused', 0), StrFixedLenField('DblSlash', '\\\\', 2), StrNullField('ServerName', 'WIN'), LEShortField('LM20Token', 65535)]

class SMBNegociate_Protocol_Request_Header_Generic(Packet):
    name = 'SMBNegociate Protocol Request Header Generic'
    fields_desc = [StrFixedLenField('Start', b'\xffSMB', 4)]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            return 10
        '\n            Depending on the first 4 bytes of the packet,\n            dispatch to the correct version of Header\n            (either SMB or SMB2)\n        '
        if _pkt and len(_pkt) >= 4:
            if _pkt[:4] == b'\xffSMB':
                return SMB_Header
            if _pkt[:4] == b'\xfeSMB':
                return SMB2_Header
        return cls
bind_layers(NBTSession, SMBNegociate_Protocol_Request_Header_Generic)