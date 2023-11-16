"""
[MS-PAC]

https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-pac/166d8064-c863-41e1-9c23-edaaa5f36962
"""
import struct
from scapy.config import conf
from scapy.error import log_runtime
from scapy.fields import ConditionalField, FieldLenField, FieldListField, FlagsField, LEIntEnumField, LELongField, LEIntField, LEShortField, MultipleTypeField, PacketLenField, PacketListField, StrField, StrFieldUtf16, StrFixedLenField, StrLenFieldUtf16, UTCTimeField, XStrField, XStrLenField
from scapy.packet import Packet
from scapy.layers.kerberos import _AUTHORIZATIONDATA_VALUES
from scapy.layers.dcerpc import _NDRConfField, NDRByteField, NDRConfStrLenField, NDRConfVarStrLenField, NDRConfVarStrLenFieldUtf16, NDRConfPacketListField, NDRConfFieldListField, NDRConfVarStrNullFieldUtf16, NDRConformantString, NDRFullPointerField, NDRInt3264EnumField, NDRIntField, NDRLongField, NDRPacket, NDRPacketField, NDRSerialization1Header, NDRShortField, NDRSignedLongField, NDRUnionField, ndr_deserialize1, ndr_serialize1
from scapy.layers.ntlm import _NTLMPayloadField, _NTLMPayloadPacket, _NTLM_post_build

class PAC_INFO_BUFFER(Packet):
    fields_desc = [LEIntEnumField('ulType', 1, {1: 'Logon information', 2: 'Credentials information', 6: 'Server checksum', 7: 'KDC checksum', 10: 'Client name and ticket information', 11: 'Constrained delegation information', 12: 'UPN and DNS information', 13: 'Client claims information', 14: 'Device information', 15: 'Device claims information', 16: 'Ticket checksum', 17: 'PAC Attributes', 18: 'PAC Requestor'}), LEIntField('cbBufferSize', None), LELongField('Offset', None)]

    def default_payload_class(self, payload):
        if False:
            return 10
        return conf.padding_layer
_PACTYPES = {}

class RPC_UNICODE_STRING(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRShortField('Length', 0), NDRShortField('MaximumLength', 0), NDRFullPointerField(NDRConfVarStrLenFieldUtf16('Buffer', '', length_from=lambda pkt: pkt.Length // 2), deferred=True)]

class FILETIME(NDRPacket):
    ALIGNMENT = (4, 4)
    fields_desc = [NDRIntField('dwLowDateTime', 0), NDRIntField('dwHighDateTime', 0)]

class PGROUP_MEMBERSHIP(NDRPacket):
    ALIGNMENT = (4, 4)
    fields_desc = [NDRIntField('RelativeId', 0), NDRIntField('Attributes', 0)]

class CYPHER_BLOCK(NDRPacket):
    fields_desc = [StrFixedLenField('data', '', length=8)]

class USER_SESSION_KEY(NDRPacket):
    fields_desc = [PacketListField('data', [], CYPHER_BLOCK, count_from=lambda _: 2)]

class RPC_SID_IDENTIFIER_AUTHORITY(NDRPacket):
    fields_desc = [StrFixedLenField('Value', '', length=6)]

class PSID(NDRPacket):
    ALIGNMENT = (4, 8)
    CONFORMANT_COUNT = 1
    fields_desc = [NDRByteField('Revision', 0), NDRByteField('SubAuthorityCount', 0), NDRPacketField('IdentifierAuthority', RPC_SID_IDENTIFIER_AUTHORITY(), RPC_SID_IDENTIFIER_AUTHORITY), NDRConfFieldListField('SubAuthority', [], NDRIntField('', 0), count_from=lambda pkt: pkt.SubAuthorityCount, conformant_in_struct=True)]

class PKERB_SID_AND_ATTRIBUTES(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRFullPointerField(NDRPacketField('Sid', PSID(), PSID), deferred=True), NDRIntField('Attributes', 0)]

class KERB_VALIDATION_INFO(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRPacketField('LogonTime', FILETIME(), FILETIME), NDRPacketField('LogoffTime', FILETIME(), FILETIME), NDRPacketField('KickOffTime', FILETIME(), FILETIME), NDRPacketField('PasswordLastSet', FILETIME(), FILETIME), NDRPacketField('PasswordCanChange', FILETIME(), FILETIME), NDRPacketField('PasswordMustChange', FILETIME(), FILETIME), NDRPacketField('EffectiveName', RPC_UNICODE_STRING(), RPC_UNICODE_STRING), NDRPacketField('FullName', RPC_UNICODE_STRING(), RPC_UNICODE_STRING), NDRPacketField('LogonScript', RPC_UNICODE_STRING(), RPC_UNICODE_STRING), NDRPacketField('ProfilePath', RPC_UNICODE_STRING(), RPC_UNICODE_STRING), NDRPacketField('HomeDirectory', RPC_UNICODE_STRING(), RPC_UNICODE_STRING), NDRPacketField('HomeDirectoryDrive', RPC_UNICODE_STRING(), RPC_UNICODE_STRING), NDRShortField('LogonCount', 0), NDRShortField('BadPasswordCount', 0), NDRIntField('UserId', 0), NDRIntField('PrimaryGroupId', 0), NDRIntField('GroupCount', 0), NDRFullPointerField(NDRConfPacketListField('GroupIds', [PGROUP_MEMBERSHIP()], PGROUP_MEMBERSHIP, count_from=lambda pkt: pkt.GroupCount), deferred=True), NDRIntField('UserFlags', 0), NDRPacketField('UserSessionKey', USER_SESSION_KEY(), USER_SESSION_KEY), NDRPacketField('LogonServer', RPC_UNICODE_STRING(), RPC_UNICODE_STRING), NDRPacketField('LogonDomainName', RPC_UNICODE_STRING(), RPC_UNICODE_STRING), NDRFullPointerField(NDRPacketField('LogonDomainId', PSID(), PSID), deferred=True), FieldListField('Reserved1', [], NDRIntField('', 0), count_from=lambda _: 2), NDRIntField('UserAccountControl', 0), FieldListField('Reserved3', [], NDRIntField('', 0), count_from=lambda _: 7), NDRIntField('SidCount', 0), NDRFullPointerField(NDRConfPacketListField('ExtraSids', [PKERB_SID_AND_ATTRIBUTES()], PKERB_SID_AND_ATTRIBUTES, count_from=lambda pkt: pkt.SidCount), deferred=True), NDRFullPointerField(NDRPacketField('ResourceGroupDomainSid', PSID(), PSID), deferred=True), NDRIntField('ResourceGroupCount', 0), NDRFullPointerField(NDRConfPacketListField('ResourceGroupIds', [PGROUP_MEMBERSHIP()], PGROUP_MEMBERSHIP, count_from=lambda pkt: pkt.ResourceGroupCount), deferred=True)]

class KERB_VALIDATION_INFO_WRAP(NDRPacket):
    fields_desc = [NDRPacketField('data', None, KERB_VALIDATION_INFO)]
_PACTYPES[1] = KERB_VALIDATION_INFO_WRAP

class PAC_CREDENTIAL_INFO(Packet):
    fields_desc = [LEIntField('Version', 0), LEIntEnumField('EncryptionType', 1, {1: 'DES-CBC-CRC', 3: 'DES-CBC-MD5', 17: 'AES128_CTS_HMAC_SHA1_96', 18: 'AES256_CTS_HMAC_SHA1_96', 23: 'RC4-HMAC'}), XStrField('SerializedData', b'')]
_PACTYPES[2] = PAC_CREDENTIAL_INFO

class PAC_CLIENT_INFO(Packet):
    fields_desc = [UTCTimeField('ClientId', None, fmt='<Q', custom_scaling=100000000.0), FieldLenField('NameLength', None, length_of='Name', fmt='<H'), StrLenFieldUtf16('Name', b'', length_from=lambda pkt: pkt.NameLength)]
_PACTYPES[10] = PAC_CLIENT_INFO

class PAC_SIGNATURE_DATA(Packet):
    fields_desc = [LEIntEnumField('SignatureType', 0, {4294967158: 'KERB_CHECKSUM_HMAC_MD5', 15: 'HMAC_SHA1_96_AES128', 16: 'HMAC_SHA1_96_AES256'}), XStrLenField('Signature', b'', length_from=lambda pkt: {4294967158: 16, 15: 12, 16: 12}.get(pkt.SignatureType, 0)), StrField('RODCIdentifier', b'')]
_PACTYPES[6] = PAC_SIGNATURE_DATA
_PACTYPES[7] = PAC_SIGNATURE_DATA
_PACTYPES[16] = PAC_SIGNATURE_DATA

class UPN_DNS_INFO(_NTLMPayloadPacket):
    fields_desc = [LEShortField('UpnLen', None), LEShortField('UpnBufferOffset', None), LEShortField('DnsDomainNameLen', None), LEShortField('DnsDomainNameBufferOffset', None), FlagsField('Flags', 0, -32, ['U', 'S']), ConditionalField(LEShortField('SamNameLen', None), lambda pkt: pkt.Flags.S), ConditionalField(LEShortField('SamNameBufferOffset', None), lambda pkt: pkt.Flags.S), ConditionalField(LEShortField('SidLen', None), lambda pkt: pkt.Flags.S), ConditionalField(LEShortField('SidBufferOffset', None), lambda pkt: pkt.Flags.S), MultipleTypeField([(_NTLMPayloadField('Payload', 20, [StrFieldUtf16('Upn', b''), StrFieldUtf16('DnsDomainName', b''), StrFieldUtf16('SamName', b''), XStrField('Sid', b'')]), lambda pkt: pkt.Flags.S)], _NTLMPayloadField('Payload', 12, [StrFieldUtf16('Upn', b''), StrFieldUtf16('DnsDomainName', b'')]))]

    def post_build(self, pkt, pay):
        if False:
            return 10
        return _NTLM_post_build(self, pkt, self.OFFSET, {'Upn': 0, 'DnsDomainName': 4}) + pay
_PACTYPES[12] = UPN_DNS_INFO
try:
    from enum import IntEnum
except ImportError:
    IntEnum = object

class CLAIM_TYPE(IntEnum):
    CLAIM_TYPE_INT64 = 1
    CLAIM_TYPE_UINT64 = 2
    CLAIM_TYPE_STRING = 3
    CLAIM_TYPE_BOOLEAN = 6

class CLAIMS_SOURCE_TYPE(IntEnum):
    CLAIMS_SOURCE_TYPE_AD = 1
    CLAIMS_SOURCE_TYPE_CERTIFICATE = 2

class CLAIMS_COMPRESSION_FORMAT(IntEnum):
    COMPRESSION_FORMAT_NONE = 0
    COMPRESSION_FORMAT_LZNT1 = 2
    COMPRESSION_FORMAT_XPRESS = 3
    COMPRESSION_FORMAT_XPRESS_HUFF = 4

class u_sub0(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRIntField('ValueCount', 0), NDRFullPointerField(NDRConfFieldListField('Int64Values', [], NDRSignedLongField, count_from=lambda pkt: pkt.ValueCount), deferred=True)]

class u_sub1(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRIntField('ValueCount', 0), NDRFullPointerField(NDRConfFieldListField('Uint64Values', [], NDRLongField, count_from=lambda pkt: pkt.ValueCount), deferred=True)]

class u_sub2(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRIntField('ValueCount', 0), NDRFullPointerField(NDRConfFieldListField('StringValues', [], NDRFullPointerField(NDRConfVarStrNullFieldUtf16('StringVal', ''), deferred=True), count_from=lambda pkt: pkt.ValueCount), deferred=True)]

class u_sub3(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRIntField('ValueCount', 0), NDRFullPointerField(NDRConfFieldListField('BooleanValues', [], NDRLongField, count_from=lambda pkt: pkt.ValueCount), deferred=True)]

class CLAIM_ENTRY(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRFullPointerField(NDRConfVarStrNullFieldUtf16('Id', ''), deferred=True), NDRInt3264EnumField('Type', 0, CLAIM_TYPE), NDRUnionField([(NDRPacketField('Values', u_sub0(), u_sub0), (lambda pkt: getattr(pkt, 'Type', None) == CLAIM_TYPE.CLAIM_TYPE_INT64, lambda _, val: val.tag == CLAIM_TYPE.CLAIM_TYPE_INT64)), (NDRPacketField('Values', u_sub1(), u_sub1), (lambda pkt: getattr(pkt, 'Type', None) == CLAIM_TYPE.CLAIM_TYPE_UINT64, lambda _, val: val.tag == CLAIM_TYPE.CLAIM_TYPE_UINT64)), (NDRPacketField('Values', u_sub2(), u_sub2), (lambda pkt: getattr(pkt, 'Type', None) == CLAIM_TYPE.CLAIM_TYPE_STRING, lambda _, val: val.tag == CLAIM_TYPE.CLAIM_TYPE_STRING)), (NDRPacketField('Values', u_sub3(), u_sub3), (lambda pkt: getattr(pkt, 'Type', None) == CLAIM_TYPE.CLAIM_TYPE_BOOLEAN, lambda _, val: val.tag == CLAIM_TYPE.CLAIM_TYPE_BOOLEAN))], StrFixedLenField('Values', '', length=0), align=(2, 8), switch_fmt=('<H', '<I'))]

class CLAIMS_ARRAY(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRInt3264EnumField('usClaimsSourceType', 0, CLAIMS_SOURCE_TYPE), NDRIntField('ulClaimsCount', 0), NDRFullPointerField(NDRConfPacketListField('ClaimEntries', [CLAIM_ENTRY()], CLAIM_ENTRY, count_from=lambda pkt: pkt.ulClaimsCount), deferred=True)]

class CLAIMS_SET(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRIntField('ulClaimsArrayCount', 0), NDRFullPointerField(NDRConfPacketListField('ClaimsArrays', [CLAIMS_ARRAY()], CLAIMS_ARRAY, count_from=lambda pkt: pkt.ulClaimsArrayCount), deferred=True), NDRShortField('usReservedType', 0), NDRIntField('ulReservedFieldSize', 0), NDRFullPointerField(NDRConfStrLenField('ReservedField', '', length_from=lambda pkt: pkt.ulReservedFieldSize), deferred=True)]

class CLAIMS_SET_WRAP(NDRPacket):
    fields_desc = [NDRPacketField('data', None, CLAIMS_SET)]

class _CLAIMSClaimSet(_NDRConfField, PacketLenField):

    def m2i(self, pkt, s):
        if False:
            while True:
                i = 10
        if pkt.usCompressionFormat == CLAIMS_COMPRESSION_FORMAT.COMPRESSION_FORMAT_NONE:
            return ndr_deserialize1(s, CLAIMS_SET_WRAP)
        else:
            return NDRConformantString(value=s)

class CLAIMS_SET_METADATA(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRIntField('ulClaimsSetSize', 0), NDRFullPointerField(_CLAIMSClaimSet('ClaimsSet', None, None, length_from=lambda pkt: pkt.ulClaimsSetSize), deferred=True), NDRInt3264EnumField('usCompressionFormat', 0, CLAIMS_COMPRESSION_FORMAT), NDRIntField('ulUncompressedClaimsSetSize', 0), NDRShortField('usReservedType', 0), NDRIntField('ulReservedFieldSize', 0), NDRFullPointerField(NDRConfVarStrLenField('ReservedField', '', length_from=lambda pkt: pkt.ulReservedFieldSize), deferred=True)]

class PAC_CLIENT_CLAIMS_INFO(NDRPacket):
    fields_desc = [NDRPacketField('Claims', None, CLAIMS_SET_METADATA)]
if IntEnum != object:
    _PACTYPES[13] = PAC_CLIENT_CLAIMS_INFO

class PDOMAIN_GROUP_MEMBERSHIP(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRFullPointerField(NDRPacketField('DomainId', PSID(), PSID), deferred=True), NDRIntField('GroupCount', 0), NDRFullPointerField(NDRConfPacketListField('GroupIds', [PGROUP_MEMBERSHIP()], PGROUP_MEMBERSHIP, count_from=lambda pkt: pkt.GroupCount), deferred=True)]

class PAC_DEVICE_INFO(NDRPacket):
    ALIGNMENT = (4, 8)
    fields_desc = [NDRIntField('UserId', 0), NDRIntField('PrimaryGroupId', 0), NDRFullPointerField(NDRPacketField('AccountDomainId', PSID(), PSID), deferred=True), NDRIntField('AccountGroupCount', 0), NDRFullPointerField(NDRConfPacketListField('AccountGroupIds', [PGROUP_MEMBERSHIP()], PGROUP_MEMBERSHIP, count_from=lambda pkt: pkt.AccountGroupCount), deferred=True), NDRIntField('SidCount', 0), NDRFullPointerField(NDRConfPacketListField('ExtraSids', [PKERB_SID_AND_ATTRIBUTES()], PKERB_SID_AND_ATTRIBUTES, count_from=lambda pkt: pkt.SidCount), deferred=True), NDRIntField('DomainGroupCount', 0), NDRFullPointerField(NDRConfPacketListField('DomainGroup', [PDOMAIN_GROUP_MEMBERSHIP()], PDOMAIN_GROUP_MEMBERSHIP, count_from=lambda pkt: pkt.DomainGroupCount), deferred=True)]

class PAC_DEVICE_INFO_WRAP(NDRPacket):
    fields_desc = [NDRPacketField('data', None, PAC_DEVICE_INFO)]
_PACTYPES[14] = PAC_DEVICE_INFO_WRAP

class PAC_ATTRIBUTES_INFO(Packet):
    fields_desc = [LEIntField('FlagsLength', 0), FieldListField('Flags', [], FlagsField('', 0, -32, {1: 'PAC_WAS_REQUESTED', 2: 'PAC_WAS_GIVEN_IMPLICITLY'}), count_from=lambda pkt: (pkt.FlagsLength + 7) // 8)]
_PACTYPES[17] = PAC_ATTRIBUTES_INFO

class PAC_REQUESTOR(Packet):
    fields_desc = [XStrField('Sid', b'')]
_PACTYPES[18] = PAC_REQUESTOR

class _PACTYPEBuffers(PacketListField):

    def addfield(self, pkt, s, val):
        if False:
            return 10
        res = b''
        if len(val) != len(pkt.Payloads):
            log_runtime.warning("Size of 'Buffers' does not match size of 'Payloads' !")
            return super(_PACTYPEBuffers, self).addfield(pkt, s, val)
        offset = 16 * len(pkt.Payloads) + 8
        for (i, v) in enumerate(val):
            x = self.i2m(pkt, v)
            lgth = len(pkt.Payloads[i])
            if v.cbBufferSize is None:
                x = x[:4] + struct.pack('<I', lgth) + x[8:]
            if v.Offset is None:
                x = x[:8] + struct.pack('<Q', offset) + x[16:]
            offset += lgth
            offset += -offset % 8
            res += x
        return s + res

class _PACTYPEPayloads(PacketListField):

    def i2m(self, pkt, val):
        if False:
            while True:
                i = 10
        if isinstance(val, NDRPacket) or isinstance(val, NDRSerialization1Header):
            s = ndr_serialize1(val)
        else:
            s = bytes(val)
        return s + b'\x00' * (-len(s) % 8)

    def getfield(self, pkt, s):
        if False:
            return 10
        if not pkt or not s:
            return (s, [])
        result = []
        for i in range(len(pkt.Buffers)):
            buf = pkt.Buffers[i]
            offset = buf.Offset - 16 * len(pkt.Buffers) - 8
            try:
                cls = _PACTYPES[buf.ulType]
                if issubclass(cls, NDRPacket):
                    val = ndr_deserialize1(s[offset:offset + buf.cbBufferSize], cls, ndr64=False)
                else:
                    val = cls(s[offset:offset + buf.cbBufferSize])
                if conf.raw_layer in val:
                    pad = conf.padding_layer(load=val[conf.raw_layer].load)
                    lay = val[conf.raw_layer].underlayer
                    lay.remove_payload()
                    lay.add_payload(pad)
            except KeyError:
                val = conf.padding_layer(s[offset:offset + buf.cbBufferSize])
            result.append(val)
        return (b'', result)

class PACTYPE(Packet):
    name = 'PACTYPE - PAC'
    fields_desc = [FieldLenField('cBuffers', None, count_of='Buffers', fmt='<I'), LEIntField('Version', 0), _PACTYPEBuffers('Buffers', [PAC_INFO_BUFFER()], PAC_INFO_BUFFER, count_from=lambda pkt: pkt.cBuffers), _PACTYPEPayloads('Payloads', [], None)]
_AUTHORIZATIONDATA_VALUES[128] = PACTYPE