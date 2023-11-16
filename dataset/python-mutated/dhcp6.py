"""
DHCPv6: Dynamic Host Configuration Protocol for IPv6. [RFC 3315,8415]
"""
import socket
import struct
import time
from scapy.ansmachine import AnsweringMachine
from scapy.arch import get_if_raw_hwaddr, in6_getifaddr
from scapy.config import conf
from scapy.data import EPOCH, ETHER_ANY
from scapy.compat import raw, orb
from scapy.error import warning
from scapy.fields import BitField, ByteEnumField, ByteField, FieldLenField, FlagsField, IntEnumField, IntField, MACField, PacketListField, ShortEnumField, ShortField, StrField, StrFixedLenField, StrLenField, UTCTimeField, X3BytesField, XIntField, XShortEnumField, PacketLenField, UUIDField, FieldListField
from scapy.data import IANA_ENTERPRISE_NUMBERS
from scapy.layers.dns import DNSStrField
from scapy.layers.inet import UDP
from scapy.layers.inet6 import DomainNameListField, IP6Field, IP6ListField, IPv6
from scapy.packet import Packet, bind_bottom_up
from scapy.pton_ntop import inet_pton
from scapy.themes import Color
from scapy.utils6 import in6_addrtovendor, in6_islladdr

def get_cls(name, fallback_cls):
    if False:
        print('Hello World!')
    return globals().get(name, fallback_cls)
dhcp6_cls_by_type = {1: 'DHCP6_Solicit', 2: 'DHCP6_Advertise', 3: 'DHCP6_Request', 4: 'DHCP6_Confirm', 5: 'DHCP6_Renew', 6: 'DHCP6_Rebind', 7: 'DHCP6_Reply', 8: 'DHCP6_Release', 9: 'DHCP6_Decline', 10: 'DHCP6_Reconf', 11: 'DHCP6_InfoRequest', 12: 'DHCP6_RelayForward', 13: 'DHCP6_RelayReply'}

def _dhcp6_dispatcher(x, *args, **kargs):
    if False:
        print('Hello World!')
    cls = conf.raw_layer
    if len(x) >= 2:
        cls = get_cls(dhcp6_cls_by_type.get(orb(x[0]), 'Raw'), conf.raw_layer)
    return cls(x, *args, **kargs)
All_DHCP_Relay_Agents_and_Servers = 'ff02::1:2'
All_DHCP_Servers = 'ff05::1:3'
dhcp6opts = {1: 'CLIENTID', 2: 'SERVERID', 3: 'IA_NA', 4: 'IA_TA', 5: 'IAADDR', 6: 'ORO', 7: 'PREFERENCE', 8: 'ELAPSED_TIME', 9: 'RELAY_MSG', 11: 'AUTH', 12: 'UNICAST', 13: 'STATUS_CODE', 14: 'RAPID_COMMIT', 15: 'USER_CLASS', 16: 'VENDOR_CLASS', 17: 'VENDOR_OPTS', 18: 'INTERFACE_ID', 19: 'RECONF_MSG', 20: 'RECONF_ACCEPT', 21: 'SIP Servers Domain Name List', 22: 'SIP Servers IPv6 Address List', 23: 'DNS Recursive Name Server Option', 24: 'Domain Search List option', 25: 'OPTION_IA_PD', 26: 'OPTION_IAPREFIX', 27: 'OPTION_NIS_SERVERS', 28: 'OPTION_NISP_SERVERS', 29: 'OPTION_NIS_DOMAIN_NAME', 30: 'OPTION_NISP_DOMAIN_NAME', 31: 'OPTION_SNTP_SERVERS', 32: 'OPTION_INFORMATION_REFRESH_TIME', 33: 'OPTION_BCMCS_SERVER_D', 34: 'OPTION_BCMCS_SERVER_A', 36: 'OPTION_GEOCONF_CIVIC', 37: 'OPTION_REMOTE_ID', 38: 'OPTION_SUBSCRIBER_ID', 39: 'OPTION_CLIENT_FQDN', 40: 'OPTION_PANA_AGENT', 41: 'OPTION_NEW_POSIX_TIMEZONE', 42: 'OPTION_NEW_TZDB_TIMEZONE', 48: 'OPTION_LQ_CLIENT_LINK', 56: 'OPTION_NTP_SERVER', 59: 'OPT_BOOTFILE_URL', 60: 'OPT_BOOTFILE_PARAM', 61: 'OPTION_CLIENT_ARCH_TYPE', 62: 'OPTION_NII', 65: 'OPTION_ERP_LOCAL_DOMAIN_NAME', 66: 'OPTION_RELAY_SUPPLIED_OPTIONS', 68: 'OPTION_VSS', 79: 'OPTION_CLIENT_LINKLAYER_ADDR', 103: 'OPTION_CAPTIVE_PORTAL', 112: 'OPTION_MUD_URL'}
dhcp6opts_by_code = {1: 'DHCP6OptClientId', 2: 'DHCP6OptServerId', 3: 'DHCP6OptIA_NA', 4: 'DHCP6OptIA_TA', 5: 'DHCP6OptIAAddress', 6: 'DHCP6OptOptReq', 7: 'DHCP6OptPref', 8: 'DHCP6OptElapsedTime', 9: 'DHCP6OptRelayMsg', 11: 'DHCP6OptAuth', 12: 'DHCP6OptServerUnicast', 13: 'DHCP6OptStatusCode', 14: 'DHCP6OptRapidCommit', 15: 'DHCP6OptUserClass', 16: 'DHCP6OptVendorClass', 17: 'DHCP6OptVendorSpecificInfo', 18: 'DHCP6OptIfaceId', 19: 'DHCP6OptReconfMsg', 20: 'DHCP6OptReconfAccept', 21: 'DHCP6OptSIPDomains', 22: 'DHCP6OptSIPServers', 23: 'DHCP6OptDNSServers', 24: 'DHCP6OptDNSDomains', 25: 'DHCP6OptIA_PD', 26: 'DHCP6OptIAPrefix', 27: 'DHCP6OptNISServers', 28: 'DHCP6OptNISPServers', 29: 'DHCP6OptNISDomain', 30: 'DHCP6OptNISPDomain', 31: 'DHCP6OptSNTPServers', 32: 'DHCP6OptInfoRefreshTime', 33: 'DHCP6OptBCMCSDomains', 34: 'DHCP6OptBCMCSServers', 37: 'DHCP6OptRemoteID', 38: 'DHCP6OptSubscriberID', 39: 'DHCP6OptClientFQDN', 40: 'DHCP6OptPanaAuthAgent', 41: 'DHCP6OptNewPOSIXTimeZone', 42: 'DHCP6OptNewTZDBTimeZone', 43: 'DHCP6OptRelayAgentERO', 48: 'DHCP6OptLQClientLink', 56: 'DHCP6OptNTPServer', 59: 'DHCP6OptBootFileUrl', 60: 'DHCP6OptBootFileParam', 61: 'DHCP6OptClientArchType', 62: 'DHCP6OptClientNetworkInterId', 65: 'DHCP6OptERPDomain', 66: 'DHCP6OptRelaySuppliedOpt', 68: 'DHCP6OptVSS', 79: 'DHCP6OptClientLinkLayerAddr', 103: 'DHCP6OptCaptivePortal', 112: 'DHCP6OptMudUrl'}
dhcp6types = {1: 'SOLICIT', 2: 'ADVERTISE', 3: 'REQUEST', 4: 'CONFIRM', 5: 'RENEW', 6: 'REBIND', 7: 'REPLY', 8: 'RELEASE', 9: 'DECLINE', 10: 'RECONFIGURE', 11: 'INFORMATION-REQUEST', 12: 'RELAY-FORW', 13: 'RELAY-REPL'}
duidtypes = {1: 'Link-layer address plus time', 2: 'Vendor-assigned unique ID based on Enterprise Number', 3: 'Link-layer Address', 4: 'UUID'}
duidhwtypes = {0: 'NET/ROM pseudo', 1: 'Ethernet (10Mb)', 2: 'Experimental Ethernet (3Mb)', 3: 'Amateur Radio AX.25', 4: 'Proteon ProNET Token Ring', 5: 'Chaos', 6: 'IEEE 802 Networks', 7: 'ARCNET', 8: 'Hyperchannel', 9: 'Lanstar', 10: 'Autonet Short Address', 11: 'LocalTalk', 12: 'LocalNet (IBM PCNet or SYTEK LocalNET)', 13: 'Ultra link', 14: 'SMDS', 15: 'Frame Relay', 16: 'Asynchronous Transmission Mode (ATM)', 17: 'HDLC', 18: 'Fibre Channel', 19: 'Asynchronous Transmission Mode (ATM)', 20: 'Serial Line', 21: 'Asynchronous Transmission Mode (ATM)', 22: 'MIL-STD-188-220', 23: 'Metricom', 24: 'IEEE 1394.1995', 25: 'MAPOS', 26: 'Twinaxial', 27: 'EUI-64', 28: 'HIPARP', 29: 'IP and ARP over ISO 7816-3', 30: 'ARPSec', 31: 'IPsec tunnel', 32: 'InfiniBand (TM)', 33: 'TIA-102 Project 25 Common Air Interface (CAI)'}

class _UTCTimeField(UTCTimeField):

    def __init__(self, *args, **kargs):
        if False:
            i = 10
            return i + 15
        epoch_2000 = (2000, 1, 1, 0, 0, 0, 5, 1, 0)
        UTCTimeField.__init__(self, *args, epoch=epoch_2000, **kargs)

class _LLAddrField(MACField):
    pass

class DUID_LLT(Packet):
    name = 'DUID - Link-layer address plus time'
    fields_desc = [ShortEnumField('type', 1, duidtypes), XShortEnumField('hwtype', 1, duidhwtypes), _UTCTimeField('timeval', 0), _LLAddrField('lladdr', ETHER_ANY)]

class DUID_EN(Packet):
    name = 'DUID - Assigned by Vendor Based on Enterprise Number'
    fields_desc = [ShortEnumField('type', 2, duidtypes), IntEnumField('enterprisenum', 311, IANA_ENTERPRISE_NUMBERS), StrField('id', '')]

class DUID_LL(Packet):
    name = 'DUID - Based on Link-layer Address'
    fields_desc = [ShortEnumField('type', 3, duidtypes), XShortEnumField('hwtype', 1, duidhwtypes), _LLAddrField('lladdr', ETHER_ANY)]

class DUID_UUID(Packet):
    name = 'DUID - Based on UUID'
    fields_desc = [ShortEnumField('type', 4, duidtypes), UUIDField('uuid', None, uuid_fmt=UUIDField.FORMAT_BE)]
duid_cls = {1: 'DUID_LLT', 2: 'DUID_EN', 3: 'DUID_LL', 4: 'DUID_UUID'}

class _DHCP6OptGuessPayload(Packet):

    @staticmethod
    def _just_guess_payload_class(cls, payload):
        if False:
            while True:
                i = 10
        if len(payload) <= 2:
            return conf.raw_layer
        opt = struct.unpack('!H', payload[:2])[0]
        clsname = dhcp6opts_by_code.get(opt, None)
        if clsname is None:
            return cls
        return get_cls(clsname, cls)

    def guess_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        return _DHCP6OptGuessPayload._just_guess_payload_class(DHCP6OptUnknown, payload)

class _DHCP6OptGuessPayloadElt(_DHCP6OptGuessPayload):
    """
    Same than _DHCP6OptGuessPayload but made for lists
    in case of list of different suboptions
    e.g. in ianaopts in DHCP6OptIA_NA
    """

    @classmethod
    def dispatch_hook(cls, payload=None, *args, **kargs):
        if False:
            while True:
                i = 10
        return cls._just_guess_payload_class(conf.raw_layer, payload)

    def extract_padding(self, s):
        if False:
            while True:
                i = 10
        return (b'', s)

class DHCP6OptUnknown(_DHCP6OptGuessPayload):
    name = 'Unknown DHCPv6 Option'
    fields_desc = [ShortEnumField('optcode', 0, dhcp6opts), FieldLenField('optlen', None, length_of='data', fmt='!H'), StrLenField('data', '', length_from=lambda pkt: pkt.optlen)]

def _duid_dispatcher(x):
    if False:
        i = 10
        return i + 15
    cls = conf.raw_layer
    if len(x) > 4:
        o = struct.unpack('!H', x[:2])[0]
        cls = get_cls(duid_cls.get(o, conf.raw_layer), conf.raw_layer)
    return cls(x)

class DHCP6OptClientId(_DHCP6OptGuessPayload):
    name = 'DHCP6 Client Identifier Option'
    fields_desc = [ShortEnumField('optcode', 1, dhcp6opts), FieldLenField('optlen', None, length_of='duid', fmt='!H'), PacketLenField('duid', '', _duid_dispatcher, length_from=lambda pkt: pkt.optlen)]

class DHCP6OptServerId(DHCP6OptClientId):
    name = 'DHCP6 Server Identifier Option'
    optcode = 2

class DHCP6OptIAAddress(_DHCP6OptGuessPayload):
    name = 'DHCP6 IA Address Option (IA_TA or IA_NA suboption)'
    fields_desc = [ShortEnumField('optcode', 5, dhcp6opts), FieldLenField('optlen', None, length_of='iaaddropts', fmt='!H', adjust=lambda pkt, x: x + 24), IP6Field('addr', '::'), IntEnumField('preflft', 0, {4294967295: 'infinity'}), IntEnumField('validlft', 0, {4294967295: 'infinity'}), PacketListField('iaaddropts', [], _DHCP6OptGuessPayloadElt, length_from=lambda pkt: pkt.optlen - 24)]

    def guess_payload_class(self, payload):
        if False:
            return 10
        return conf.padding_layer

class DHCP6OptIA_NA(_DHCP6OptGuessPayload):
    name = 'DHCP6 Identity Association for Non-temporary Addresses Option'
    fields_desc = [ShortEnumField('optcode', 3, dhcp6opts), FieldLenField('optlen', None, length_of='ianaopts', fmt='!H', adjust=lambda pkt, x: x + 12), XIntField('iaid', None), IntField('T1', None), IntField('T2', None), PacketListField('ianaopts', [], _DHCP6OptGuessPayloadElt, length_from=lambda pkt: pkt.optlen - 12)]

class DHCP6OptIA_TA(_DHCP6OptGuessPayload):
    name = 'DHCP6 Identity Association for Temporary Addresses Option'
    fields_desc = [ShortEnumField('optcode', 4, dhcp6opts), FieldLenField('optlen', None, length_of='iataopts', fmt='!H', adjust=lambda pkt, x: x + 4), XIntField('iaid', None), PacketListField('iataopts', [], _DHCP6OptGuessPayloadElt, length_from=lambda pkt: pkt.optlen - 4)]

class _OptReqListField(StrLenField):
    islist = 1

    def i2h(self, pkt, x):
        if False:
            i = 10
            return i + 15
        if x is None:
            return []
        return x

    def i2len(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        return 2 * len(x)

    def any2i(self, pkt, x):
        if False:
            while True:
                i = 10
        return x

    def i2repr(self, pkt, x):
        if False:
            print('Hello World!')
        s = []
        for y in self.i2h(pkt, x):
            if y in dhcp6opts:
                s.append(dhcp6opts[y])
            else:
                s.append('%d' % y)
        return '[%s]' % ', '.join(s)

    def m2i(self, pkt, x):
        if False:
            print('Hello World!')
        r = []
        while len(x) != 0:
            if len(x) < 2:
                warning('Odd length for requested option field. Rejecting last byte')
                return r
            r.append(struct.unpack('!H', x[:2])[0])
            x = x[2:]
        return r

    def i2m(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        return b''.join((struct.pack('!H', y) for y in x))

class DHCP6OptOptReq(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option Request Option'
    fields_desc = [ShortEnumField('optcode', 6, dhcp6opts), FieldLenField('optlen', None, length_of='reqopts', fmt='!H'), _OptReqListField('reqopts', [23, 24], length_from=lambda pkt: pkt.optlen)]

class DHCP6OptPref(_DHCP6OptGuessPayload):
    name = 'DHCP6 Preference Option'
    fields_desc = [ShortEnumField('optcode', 7, dhcp6opts), ShortField('optlen', 1), ByteField('prefval', 255)]

class _ElapsedTimeField(ShortField):

    def i2repr(self, pkt, x):
        if False:
            print('Hello World!')
        if x == 65535:
            return 'infinity (0xffff)'
        return '%.2f sec' % (self.i2h(pkt, x) / 100.0)

class DHCP6OptElapsedTime(_DHCP6OptGuessPayload):
    name = 'DHCP6 Elapsed Time Option'
    fields_desc = [ShortEnumField('optcode', 8, dhcp6opts), ShortField('optlen', 2), _ElapsedTimeField('elapsedtime', 0)]
_dhcp6_auth_proto = {0: 'configuration token', 1: 'delayed authentication', 2: 'delayed authentication (obsolete)', 3: 'reconfigure key'}
_dhcp6_auth_alg = {0: 'configuration token', 1: 'HMAC-MD5'}
_dhcp6_auth_rdm = {0: 'use of a monotonically increasing value'}

class DHCP6OptAuth(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - Authentication'
    fields_desc = [ShortEnumField('optcode', 11, dhcp6opts), FieldLenField('optlen', None, length_of='authinfo', fmt='!H', adjust=lambda pkt, x: x + 11), ByteEnumField('proto', 3, _dhcp6_auth_proto), ByteEnumField('alg', 1, _dhcp6_auth_alg), ByteEnumField('rdm', 0, _dhcp6_auth_rdm), StrFixedLenField('replay', b'\x00' * 8, 8), StrLenField('authinfo', '', length_from=lambda pkt: pkt.optlen - 11)]

class _SrvAddrField(IP6Field):

    def i2h(self, pkt, x):
        if False:
            print('Hello World!')
        if x is None:
            return '::'
        return x

    def i2m(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        return inet_pton(socket.AF_INET6, self.i2h(pkt, x))

class DHCP6OptServerUnicast(_DHCP6OptGuessPayload):
    name = 'DHCP6 Server Unicast Option'
    fields_desc = [ShortEnumField('optcode', 12, dhcp6opts), ShortField('optlen', 16), _SrvAddrField('srvaddr', None)]
dhcp6statuscodes = {0: 'Success', 1: 'UnspecFail', 2: 'NoAddrsAvail', 3: 'NoBinding', 4: 'NotOnLink', 5: 'UseMulticast', 6: 'NoPrefixAvail'}

class DHCP6OptStatusCode(_DHCP6OptGuessPayload):
    name = 'DHCP6 Status Code Option'
    fields_desc = [ShortEnumField('optcode', 13, dhcp6opts), FieldLenField('optlen', None, length_of='statusmsg', fmt='!H', adjust=lambda pkt, x: x + 2), ShortEnumField('statuscode', None, dhcp6statuscodes), StrLenField('statusmsg', '', length_from=lambda pkt: pkt.optlen - 2)]

class DHCP6OptRapidCommit(_DHCP6OptGuessPayload):
    name = 'DHCP6 Rapid Commit Option'
    fields_desc = [ShortEnumField('optcode', 14, dhcp6opts), ShortField('optlen', 0)]

class _UserClassDataField(PacketListField):

    def i2len(self, pkt, z):
        if False:
            while True:
                i = 10
        if z is None or z == []:
            return 0
        return sum((len(raw(x)) for x in z))

    def getfield(self, pkt, s):
        if False:
            return 10
        tmp_len = self.length_from(pkt)
        lst = []
        (remain, payl) = (s[:tmp_len], s[tmp_len:])
        while len(remain) > 0:
            p = self.m2i(pkt, remain)
            if conf.padding_layer in p:
                pad = p[conf.padding_layer]
                remain = pad.load
                del pad.underlayer.payload
            else:
                remain = ''
            lst.append(p)
        return (payl, lst)

class USER_CLASS_DATA(Packet):
    name = 'user class data'
    fields_desc = [FieldLenField('len', None, length_of='data'), StrLenField('data', '', length_from=lambda pkt: pkt.len)]

    def guess_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        return conf.padding_layer

class DHCP6OptUserClass(_DHCP6OptGuessPayload):
    name = 'DHCP6 User Class Option'
    fields_desc = [ShortEnumField('optcode', 15, dhcp6opts), FieldLenField('optlen', None, fmt='!H', length_of='userclassdata'), _UserClassDataField('userclassdata', [], USER_CLASS_DATA, length_from=lambda pkt: pkt.optlen)]

class _VendorClassDataField(_UserClassDataField):
    pass

class VENDOR_CLASS_DATA(USER_CLASS_DATA):
    name = 'vendor class data'

class DHCP6OptVendorClass(_DHCP6OptGuessPayload):
    name = 'DHCP6 Vendor Class Option'
    fields_desc = [ShortEnumField('optcode', 16, dhcp6opts), FieldLenField('optlen', None, length_of='vcdata', fmt='!H', adjust=lambda pkt, x: x + 4), IntEnumField('enterprisenum', None, IANA_ENTERPRISE_NUMBERS), _VendorClassDataField('vcdata', [], VENDOR_CLASS_DATA, length_from=lambda pkt: pkt.optlen - 4)]

class VENDOR_SPECIFIC_OPTION(_DHCP6OptGuessPayload):
    name = 'vendor specific option data'
    fields_desc = [ShortField('optcode', None), FieldLenField('optlen', None, length_of='optdata'), StrLenField('optdata', '', length_from=lambda pkt: pkt.optlen)]

    def guess_payload_class(self, payload):
        if False:
            while True:
                i = 10
        return conf.padding_layer

class DHCP6OptVendorSpecificInfo(_DHCP6OptGuessPayload):
    name = 'DHCP6 Vendor-specific Information Option'
    fields_desc = [ShortEnumField('optcode', 17, dhcp6opts), FieldLenField('optlen', None, length_of='vso', fmt='!H', adjust=lambda pkt, x: x + 4), IntEnumField('enterprisenum', None, IANA_ENTERPRISE_NUMBERS), _VendorClassDataField('vso', [], VENDOR_SPECIFIC_OPTION, length_from=lambda pkt: pkt.optlen - 4)]

class DHCP6OptIfaceId(_DHCP6OptGuessPayload):
    name = 'DHCP6 Interface-Id Option'
    fields_desc = [ShortEnumField('optcode', 18, dhcp6opts), FieldLenField('optlen', None, fmt='!H', length_of='ifaceid'), StrLenField('ifaceid', '', length_from=lambda pkt: pkt.optlen)]

class DHCP6OptReconfMsg(_DHCP6OptGuessPayload):
    name = 'DHCP6 Reconfigure Message Option'
    fields_desc = [ShortEnumField('optcode', 19, dhcp6opts), ShortField('optlen', 1), ByteEnumField('msgtype', 11, {5: 'Renew Message', 11: 'Information Request'})]

class DHCP6OptReconfAccept(_DHCP6OptGuessPayload):
    name = 'DHCP6 Reconfigure Accept Option'
    fields_desc = [ShortEnumField('optcode', 20, dhcp6opts), ShortField('optlen', 0)]

class DHCP6OptSIPDomains(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - SIP Servers Domain Name List'
    fields_desc = [ShortEnumField('optcode', 21, dhcp6opts), FieldLenField('optlen', None, length_of='sipdomains'), DomainNameListField('sipdomains', [], length_from=lambda pkt: pkt.optlen)]

class DHCP6OptSIPServers(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - SIP Servers IPv6 Address List'
    fields_desc = [ShortEnumField('optcode', 22, dhcp6opts), FieldLenField('optlen', None, length_of='sipservers'), IP6ListField('sipservers', [], length_from=lambda pkt: pkt.optlen)]

class DHCP6OptDNSServers(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - DNS Recursive Name Server'
    fields_desc = [ShortEnumField('optcode', 23, dhcp6opts), FieldLenField('optlen', None, length_of='dnsservers'), IP6ListField('dnsservers', [], length_from=lambda pkt: pkt.optlen)]

class DHCP6OptDNSDomains(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - Domain Search List option'
    fields_desc = [ShortEnumField('optcode', 24, dhcp6opts), FieldLenField('optlen', None, length_of='dnsdomains'), DomainNameListField('dnsdomains', [], length_from=lambda pkt: pkt.optlen)]

class DHCP6OptIAPrefix(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - IA Prefix option'
    fields_desc = [ShortEnumField('optcode', 26, dhcp6opts), FieldLenField('optlen', None, length_of='iaprefopts', adjust=lambda pkt, x: x + 25), IntEnumField('preflft', 0, {4294967295: 'infinity'}), IntEnumField('validlft', 0, {4294967295: 'infinity'}), ByteField('plen', 48), IP6Field('prefix', '2001:db8::'), PacketListField('iaprefopts', [], _DHCP6OptGuessPayloadElt, length_from=lambda pkt: pkt.optlen - 25)]

    def guess_payload_class(self, payload):
        if False:
            return 10
        return conf.padding_layer

class DHCP6OptIA_PD(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - Identity Association for Prefix Delegation'
    fields_desc = [ShortEnumField('optcode', 25, dhcp6opts), FieldLenField('optlen', None, length_of='iapdopt', fmt='!H', adjust=lambda pkt, x: x + 12), XIntField('iaid', None), IntField('T1', None), IntField('T2', None), PacketListField('iapdopt', [], _DHCP6OptGuessPayloadElt, length_from=lambda pkt: pkt.optlen - 12)]

class DHCP6OptNISServers(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - NIS Servers'
    fields_desc = [ShortEnumField('optcode', 27, dhcp6opts), FieldLenField('optlen', None, length_of='nisservers'), IP6ListField('nisservers', [], length_from=lambda pkt: pkt.optlen)]

class DHCP6OptNISPServers(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - NIS+ Servers'
    fields_desc = [ShortEnumField('optcode', 28, dhcp6opts), FieldLenField('optlen', None, length_of='nispservers'), IP6ListField('nispservers', [], length_from=lambda pkt: pkt.optlen)]

class DHCP6OptNISDomain(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - NIS Domain Name'
    fields_desc = [ShortEnumField('optcode', 29, dhcp6opts), FieldLenField('optlen', None, length_of='nisdomain'), DNSStrField('nisdomain', '', length_from=lambda pkt: pkt.optlen)]

class DHCP6OptNISPDomain(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - NIS+ Domain Name'
    fields_desc = [ShortEnumField('optcode', 30, dhcp6opts), FieldLenField('optlen', None, length_of='nispdomain'), DNSStrField('nispdomain', '', length_from=lambda pkt: pkt.optlen)]

class DHCP6OptSNTPServers(_DHCP6OptGuessPayload):
    name = 'DHCP6 option - SNTP Servers'
    fields_desc = [ShortEnumField('optcode', 31, dhcp6opts), FieldLenField('optlen', None, length_of='sntpservers'), IP6ListField('sntpservers', [], length_from=lambda pkt: pkt.optlen)]
IRT_DEFAULT = 86400
IRT_MINIMUM = 600

class DHCP6OptInfoRefreshTime(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - Information Refresh Time'
    fields_desc = [ShortEnumField('optcode', 32, dhcp6opts), ShortField('optlen', 4), IntField('reftime', IRT_DEFAULT)]

class DHCP6OptBCMCSDomains(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - BCMCS Domain Name List'
    fields_desc = [ShortEnumField('optcode', 33, dhcp6opts), FieldLenField('optlen', None, length_of='bcmcsdomains'), DomainNameListField('bcmcsdomains', [], length_from=lambda pkt: pkt.optlen)]

class DHCP6OptBCMCSServers(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - BCMCS Addresses List'
    fields_desc = [ShortEnumField('optcode', 34, dhcp6opts), FieldLenField('optlen', None, length_of='bcmcsservers'), IP6ListField('bcmcsservers', [], length_from=lambda pkt: pkt.optlen)]
_dhcp6_geoconf_what = {0: 'DHCP server', 1: 'closest network element', 2: 'client'}

class DHCP6OptGeoConfElement(Packet):
    fields_desc = [ByteField('CAtype', 0), FieldLenField('CAlength', None, length_of='CAvalue'), StrLenField('CAvalue', '', length_from=lambda pkt: pkt.CAlength)]

class DHCP6OptGeoConf(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - Civic Location'
    fields_desc = [ShortEnumField('optcode', 36, dhcp6opts), FieldLenField('optlen', None, length_of='ca_elts', adjust=lambda x: x + 3), ByteEnumField('what', 2, _dhcp6_geoconf_what), StrFixedLenField('country_code', 'FR', 2), PacketListField('ca_elts', [], DHCP6OptGeoConfElement, length_from=lambda pkt: pkt.optlen - 3)]

class DHCP6OptRemoteID(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - Relay Agent Remote-ID'
    fields_desc = [ShortEnumField('optcode', 37, dhcp6opts), FieldLenField('optlen', None, length_of='remoteid', adjust=lambda pkt, x: x + 4), IntEnumField('enterprisenum', None, IANA_ENTERPRISE_NUMBERS), StrLenField('remoteid', '', length_from=lambda pkt: pkt.optlen - 4)]

class DHCP6OptSubscriberID(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - Subscriber ID'
    fields_desc = [ShortEnumField('optcode', 38, dhcp6opts), FieldLenField('optlen', None, length_of='subscriberid'), StrLenField('subscriberid', '', length_from=lambda pkt: pkt.optlen)]

class DHCP6OptClientFQDN(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - Client FQDN'
    fields_desc = [ShortEnumField('optcode', 39, dhcp6opts), FieldLenField('optlen', None, length_of='fqdn', adjust=lambda pkt, x: x + 1), BitField('res', 0, 5), FlagsField('flags', 0, 3, 'SON'), DNSStrField('fqdn', '', length_from=lambda pkt: pkt.optlen - 1)]

class DHCP6OptPanaAuthAgent(_DHCP6OptGuessPayload):
    name = 'DHCP6 PANA Authentication Agent Option'
    fields_desc = [ShortEnumField('optcode', 40, dhcp6opts), FieldLenField('optlen', None, length_of='paaaddr'), IP6ListField('paaaddr', [], length_from=lambda pkt: pkt.optlen)]

class DHCP6OptNewPOSIXTimeZone(_DHCP6OptGuessPayload):
    name = 'DHCP6 POSIX Timezone Option'
    fields_desc = [ShortEnumField('optcode', 41, dhcp6opts), FieldLenField('optlen', None, length_of='optdata'), StrLenField('optdata', '', length_from=lambda pkt: pkt.optlen)]

class DHCP6OptNewTZDBTimeZone(_DHCP6OptGuessPayload):
    name = 'DHCP6 TZDB Timezone Option'
    fields_desc = [ShortEnumField('optcode', 42, dhcp6opts), FieldLenField('optlen', None, length_of='optdata'), StrLenField('optdata', '', length_from=lambda pkt: pkt.optlen)]

class DHCP6OptRelayAgentERO(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - RelayRequest Option'
    fields_desc = [ShortEnumField('optcode', 43, dhcp6opts), FieldLenField('optlen', None, length_of='reqopts', fmt='!H'), _OptReqListField('reqopts', [23, 24], length_from=lambda pkt: pkt.optlen)]

class DHCP6OptLQClientLink(_DHCP6OptGuessPayload):
    name = 'DHCP6 Client Link Option'
    fields_desc = [ShortEnumField('optcode', 48, dhcp6opts), FieldLenField('optlen', None, length_of='linkaddress'), IP6ListField('linkaddress', [], length_from=lambda pkt: pkt.optlen)]

class DHCP6NTPSubOptSrvAddr(Packet):
    name = 'DHCP6 NTP Server Address Suboption'
    fields_desc = [ShortField('optcode', 1), ShortField('optlen', 16), IP6Field('addr', '::')]

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        return (b'', s)

class DHCP6NTPSubOptMCAddr(Packet):
    name = 'DHCP6 NTP Multicast Address Suboption'
    fields_desc = [ShortField('optcode', 2), ShortField('optlen', 16), IP6Field('addr', '::')]

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        return (b'', s)

class DHCP6NTPSubOptSrvFQDN(Packet):
    name = 'DHCP6 NTP Server FQDN Suboption'
    fields_desc = [ShortField('optcode', 3), FieldLenField('optlen', None, length_of='fqdn'), DNSStrField('fqdn', '', length_from=lambda pkt: pkt.optlen)]

    def extract_padding(self, s):
        if False:
            return 10
        return (b'', s)
_ntp_subopts = {1: DHCP6NTPSubOptSrvAddr, 2: DHCP6NTPSubOptMCAddr, 3: DHCP6NTPSubOptSrvFQDN}

def _ntp_subopt_dispatcher(p, **kwargs):
    if False:
        return 10
    cls = conf.raw_layer
    if len(p) >= 2:
        o = struct.unpack('!H', p[:2])[0]
        cls = _ntp_subopts.get(o, conf.raw_layer)
    return cls(p, **kwargs)

class DHCP6OptNTPServer(_DHCP6OptGuessPayload):
    name = 'DHCP6 NTP Server Option'
    fields_desc = [ShortEnumField('optcode', 56, dhcp6opts), FieldLenField('optlen', None, length_of='ntpserver', fmt='!H'), PacketListField('ntpserver', [], _ntp_subopt_dispatcher, length_from=lambda pkt: pkt.optlen)]

class DHCP6OptBootFileUrl(_DHCP6OptGuessPayload):
    name = 'DHCP6 Boot File URL Option'
    fields_desc = [ShortEnumField('optcode', 59, dhcp6opts), FieldLenField('optlen', None, length_of='optdata'), StrLenField('optdata', '', length_from=lambda pkt: pkt.optlen)]

class DHCP6OptClientArchType(_DHCP6OptGuessPayload):
    name = 'DHCP6 Client System Architecture Type Option'
    fields_desc = [ShortEnumField('optcode', 61, dhcp6opts), FieldLenField('optlen', None, length_of='archtypes', fmt='!H'), FieldListField('archtypes', [], ShortField('archtype', 0), length_from=lambda pkt: pkt.optlen)]

class DHCP6OptClientNetworkInterId(_DHCP6OptGuessPayload):
    name = 'DHCP6 Client Network Interface Identifier Option'
    fields_desc = [ShortEnumField('optcode', 62, dhcp6opts), ShortField('optlen', 3), ByteField('iitype', 0), ByteField('iimajor', 0), ByteField('iiminor', 0)]

class DHCP6OptERPDomain(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - ERP Domain Name List'
    fields_desc = [ShortEnumField('optcode', 65, dhcp6opts), FieldLenField('optlen', None, length_of='erpdomain'), DomainNameListField('erpdomain', [], length_from=lambda pkt: pkt.optlen)]

class DHCP6OptRelaySuppliedOpt(_DHCP6OptGuessPayload):
    name = 'DHCP6 Relay-Supplied Options Option'
    fields_desc = [ShortEnumField('optcode', 66, dhcp6opts), FieldLenField('optlen', None, length_of='relaysupplied', fmt='!H'), PacketListField('relaysupplied', [], _DHCP6OptGuessPayloadElt, length_from=lambda pkt: pkt.optlen)]

class DHCP6OptVSS(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - Virtual Subnet Selection'
    fields_desc = [ShortEnumField('optcode', 68, dhcp6opts), FieldLenField('optlen', None, length_of='data', adjust=lambda pkt, x: x + 1), ByteField('type', 255), StrLenField('data', '', length_from=lambda pkt: pkt.optlen)]

class DHCP6OptClientLinkLayerAddr(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - Client Link Layer address'
    fields_desc = [ShortEnumField('optcode', 79, dhcp6opts), FieldLenField('optlen', None, length_of='clladdr', adjust=lambda pkt, x: x + 2), ShortField('lltype', 1), _LLAddrField('clladdr', ETHER_ANY)]

class DHCP6OptCaptivePortal(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - Captive-Portal'
    fields_desc = [ShortEnumField('optcode', 103, dhcp6opts), FieldLenField('optlen', None, length_of='URI'), StrLenField('URI', '', length_from=lambda pkt: pkt.optlen)]

class DHCP6OptMudUrl(_DHCP6OptGuessPayload):
    name = 'DHCP6 Option - MUD URL'
    fields_desc = [ShortEnumField('optcode', 112, dhcp6opts), FieldLenField('optlen', None, length_of='mudstring'), StrLenField('mudstring', '', length_from=lambda pkt: pkt.optlen, max_length=253)]
DHCP6RelayAgentUnicastAddr = ''
DHCP6RelayHopCount = ''
DHCP6ServerUnicastAddr = ''
DHCP6ClientUnicastAddr = ''
DHCP6ClientIA_TA = ''
DHCP6ClientIA_NA = ''
DHCP6ClientIAID = ''
T1 = ''
T2 = ''
DHCP6ServerDUID = ''
DHCP6CurrentTransactionID = ''
DHCP6PrefVal = ''

class DHCP6(_DHCP6OptGuessPayload):
    name = 'DHCPv6 Generic Message'
    fields_desc = [ByteEnumField('msgtype', None, dhcp6types), X3BytesField('trid', 0)]
    overload_fields = {UDP: {'sport': 546, 'dport': 547}}

    def hashret(self):
        if False:
            for i in range(10):
                print('nop')
        return struct.pack('!I', self.trid)[1:4]

class DHCP6OptRelayMsg(_DHCP6OptGuessPayload):
    name = 'DHCP6 Relay Message Option'
    fields_desc = [ShortEnumField('optcode', 9, dhcp6opts), FieldLenField('optlen', None, fmt='!H', length_of='message'), PacketLenField('message', DHCP6(), _dhcp6_dispatcher, length_from=lambda p: p.optlen)]

class DHCP6_Solicit(DHCP6):
    name = 'DHCPv6 Solicit Message'
    msgtype = 1
    overload_fields = {UDP: {'sport': 546, 'dport': 547}}

class DHCP6_Advertise(DHCP6):
    name = 'DHCPv6 Advertise Message'
    msgtype = 2
    overload_fields = {UDP: {'sport': 547, 'dport': 546}}

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, DHCP6_Solicit) and other.msgtype == 1 and (self.trid == other.trid)

class DHCP6_Request(DHCP6):
    name = 'DHCPv6 Request Message'
    msgtype = 3

class DHCP6_Confirm(DHCP6):
    name = 'DHCPv6 Confirm Message'
    msgtype = 4

class DHCP6_Renew(DHCP6):
    name = 'DHCPv6 Renew Message'
    msgtype = 5

class DHCP6_Rebind(DHCP6):
    name = 'DHCPv6 Rebind Message'
    msgtype = 6

class DHCP6_Reply(DHCP6):
    name = 'DHCPv6 Reply Message'
    msgtype = 7
    overload_fields = {UDP: {'sport': 547, 'dport': 546}}

    def answers(self, other):
        if False:
            return 10
        types = (DHCP6_Solicit, DHCP6_InfoRequest, DHCP6_Confirm, DHCP6_Rebind, DHCP6_Decline, DHCP6_Request, DHCP6_Release, DHCP6_Renew)
        return isinstance(other, types) and self.trid == other.trid

class DHCP6_Release(DHCP6):
    name = 'DHCPv6 Release Message'
    msgtype = 8

class DHCP6_Decline(DHCP6):
    name = 'DHCPv6 Decline Message'
    msgtype = 9

class DHCP6_Reconf(DHCP6):
    name = 'DHCPv6 Reconfigure Message'
    msgtype = 10
    overload_fields = {UDP: {'sport': 547, 'dport': 546}}

class DHCP6_InfoRequest(DHCP6):
    name = 'DHCPv6 Information Request Message'
    msgtype = 11

class DHCP6_RelayForward(_DHCP6OptGuessPayload, Packet):
    name = 'DHCPv6 Relay Forward Message (Relay Agent/Server Message)'
    fields_desc = [ByteEnumField('msgtype', 12, dhcp6types), ByteField('hopcount', None), IP6Field('linkaddr', '::'), IP6Field('peeraddr', '::')]
    overload_fields = {UDP: {'sport': 547, 'dport': 547}}

    def hashret(self):
        if False:
            for i in range(10):
                print('nop')
        return inet_pton(socket.AF_INET6, self.peeraddr)

class DHCP6_RelayReply(DHCP6_RelayForward):
    name = 'DHCPv6 Relay Reply Message (Relay Agent/Server Message)'
    msgtype = 13

    def hashret(self):
        if False:
            print('Hello World!')
        return inet_pton(socket.AF_INET6, self.peeraddr)

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, DHCP6_RelayForward) and self.hopcount == other.hopcount and (self.linkaddr == other.linkaddr) and (self.peeraddr == other.peeraddr)
bind_bottom_up(UDP, _dhcp6_dispatcher, {'dport': 547})
bind_bottom_up(UDP, _dhcp6_dispatcher, {'dport': 546})

class DHCPv6_am(AnsweringMachine):
    function_name = 'dhcp6d'
    filter = 'udp and port 546 and port 547'

    def usage(self):
        if False:
            print('Hello World!')
        msg = '\nDHCPv6_am.parse_options( dns="2001:500::1035", domain="localdomain, local",\n        duid=None, iface=conf.iface, advpref=255, sntpservers=None,\n        sipdomains=None, sipservers=None,\n        nisdomain=None, nisservers=None,\n        nispdomain=None, nispservers=None,\n        bcmcsdomains=None, bcmcsservers=None)\n\n   debug : When set, additional debugging information is printed.\n\n   duid   : some DUID class (DUID_LLT, DUID_LL or DUID_EN). If none\n            is provided a DUID_LLT is constructed based on the MAC\n            address of the sending interface and launch time of dhcp6d\n            answering machine.\n\n   iface : the interface to listen/reply on if you do not want to use\n           conf.iface.\n\n   advpref : Value in [0,255] given to Advertise preference field.\n             By default, 255 is used. Be aware that this specific\n             value makes clients stops waiting for further Advertise\n             messages from other servers.\n\n   dns : list of recursive DNS servers addresses (as a string or list).\n         By default, it is set empty and the associated DHCP6OptDNSServers\n         option is inactive. See RFC 3646 for details.\n   domain : a list of DNS search domain (as a string or list). By default,\n         it is empty and the associated DHCP6OptDomains option is inactive.\n         See RFC 3646 for details.\n\n   sntpservers : a list of SNTP servers IPv6 addresses. By default,\n         it is empty and the associated DHCP6OptSNTPServers option\n         is inactive.\n\n   sipdomains : a list of SIP domains. By default, it is empty and the\n         associated DHCP6OptSIPDomains option is inactive. See RFC 3319\n         for details.\n   sipservers : a list of SIP servers IPv6 addresses. By default, it is\n         empty and the associated DHCP6OptSIPDomains option is inactive.\n         See RFC 3319 for details.\n\n   nisdomain : a list of NIS domains. By default, it is empty and the\n         associated DHCP6OptNISDomains option is inactive. See RFC 3898\n         for details. See RFC 3646 for details.\n   nisservers : a list of NIS servers IPv6 addresses. By default, it is\n         empty and the associated DHCP6OptNISServers option is inactive.\n         See RFC 3646 for details.\n\n   nispdomain : a list of NIS+ domains. By default, it is empty and the\n         associated DHCP6OptNISPDomains option is inactive. See RFC 3898\n         for details.\n   nispservers : a list of NIS+ servers IPv6 addresses. By default, it is\n         empty and the associated DHCP6OptNISServers option is inactive.\n         See RFC 3898 for details.\n\n   bcmcsdomain : a list of BCMCS domains. By default, it is empty and the\n         associated DHCP6OptBCMCSDomains option is inactive. See RFC 4280\n         for details.\n   bcmcsservers : a list of BCMCS servers IPv6 addresses. By default, it is\n         empty and the associated DHCP6OptBCMCSServers option is inactive.\n         See RFC 4280 for details.\n\n   If you have a need for others, just ask ... or provide a patch.'
        print(msg)

    def parse_options(self, dns='2001:500::1035', domain='localdomain, local', startip='2001:db8::1', endip='2001:db8::20', duid=None, sntpservers=None, sipdomains=None, sipservers=None, nisdomain=None, nisservers=None, nispdomain=None, nispservers=None, bcmcsservers=None, bcmcsdomains=None, iface=None, debug=0, advpref=255):
        if False:
            print('Hello World!')

        def norm_list(val, param_name):
            if False:
                return 10
            if val is None:
                return None
            if isinstance(val, list):
                return val
            elif isinstance(val, str):
                tmp_len = val.split(',')
                return [x.strip() for x in tmp_len]
            else:
                print("Bad '%s' parameter provided." % param_name)
                self.usage()
                return -1
        if iface is None:
            iface = conf.iface
        self.debug = debug
        self.dhcpv6_options = {}
        for o in [(dns, 'dns', 23, lambda x: DHCP6OptDNSServers(dnsservers=x)), (domain, 'domain', 24, lambda x: DHCP6OptDNSDomains(dnsdomains=x)), (sntpservers, 'sntpservers', 31, lambda x: DHCP6OptSNTPServers(sntpservers=x)), (sipservers, 'sipservers', 22, lambda x: DHCP6OptSIPServers(sipservers=x)), (sipdomains, 'sipdomains', 21, lambda x: DHCP6OptSIPDomains(sipdomains=x)), (nisservers, 'nisservers', 27, lambda x: DHCP6OptNISServers(nisservers=x)), (nisdomain, 'nisdomain', 29, lambda x: DHCP6OptNISDomain(nisdomain=(x + [''])[0])), (nispservers, 'nispservers', 28, lambda x: DHCP6OptNISPServers(nispservers=x)), (nispdomain, 'nispdomain', 30, lambda x: DHCP6OptNISPDomain(nispdomain=(x + [''])[0])), (bcmcsservers, 'bcmcsservers', 33, lambda x: DHCP6OptBCMCSServers(bcmcsservers=x)), (bcmcsdomains, 'bcmcsdomains', 34, lambda x: DHCP6OptBCMCSDomains(bcmcsdomains=x))]:
            opt = norm_list(o[0], o[1])
            if opt == -1:
                return False
            elif opt is None:
                pass
            else:
                self.dhcpv6_options[o[2]] = o[3](opt)
        if self.debug:
            print('\n[+] List of active DHCPv6 options:')
            opts = sorted(self.dhcpv6_options)
            for i in opts:
                print('    %d: %s' % (i, repr(self.dhcpv6_options[i])))
        self.advpref = advpref
        self.startip = startip
        self.endip = endip
        self.iface = iface
        if duid is not None:
            self.duid = duid
        else:
            epoch = (2000, 1, 1, 0, 0, 0, 5, 1, 0)
            delta = time.mktime(epoch) - EPOCH
            timeval = time.time() - delta
            rawmac = get_if_raw_hwaddr(iface)[1]
            mac = ':'.join(('%.02x' % orb(x) for x in rawmac))
            self.duid = DUID_LLT(timeval=timeval, lladdr=mac)
        if self.debug:
            print('\n[+] Our server DUID:')
            self.duid.show(label_lvl=' ' * 4)
        self.src_addr = None
        try:
            addr = next((x for x in in6_getifaddr() if x[2] == iface and in6_islladdr(x[0])))
        except (StopIteration, RuntimeError):
            warning('Unable to get a Link-Local address')
            return
        else:
            self.src_addr = addr[0]
        self.leases = {}
        if self.debug:
            print('\n[+] Starting DHCPv6 service on %s:' % self.iface)

    def is_request(self, p):
        if False:
            i = 10
            return i + 15
        if IPv6 not in p:
            return False
        src = p[IPv6].src
        p = p[IPv6].payload
        if not isinstance(p, UDP) or p.sport != 546 or p.dport != 547:
            return False
        p = p.payload
        if not isinstance(p, DHCP6):
            return False
        if not p.msgtype in [1, 3, 4, 5, 6, 8, 9, 11]:
            return False
        if p.msgtype == 1 or p.msgtype == 6 or p.msgtype == 4:
            if DHCP6OptClientId not in p or DHCP6OptServerId in p:
                return False
            if p.msgtype == 6 or p.msgtype == 4:
                return False
        elif p.msgtype == 3 or p.msgtype == 5 or p.msgtype == 8:
            if DHCP6OptServerId not in p or DHCP6OptClientId not in p:
                return False
            duid = p[DHCP6OptServerId].duid
            if not isinstance(duid, type(self.duid)):
                return False
            if raw(duid) != raw(self.duid):
                return False
            if p.msgtype == 5 or p.msgtype == 8:
                return False
        elif p.msgtype == 9:
            if not self.debug:
                return False
            bo = Color.bold
            g = Color.green + bo
            b = Color.blue + bo
            n = Color.normal
            r = Color.red
            vendor = in6_addrtovendor(src)
            if vendor and vendor != 'UNKNOWN':
                vendor = ' [' + b + vendor + n + ']'
            else:
                vendor = ''
            src = bo + src + n
            it = p
            addrs = []
            while it:
                lst = []
                if isinstance(it, DHCP6OptIA_NA):
                    lst = it.ianaopts
                elif isinstance(it, DHCP6OptIA_TA):
                    lst = it.iataopts
                addrs += [x.addr for x in lst if isinstance(x, DHCP6OptIAAddress)]
                it = it.payload
            addrs = [bo + x + n for x in addrs]
            if self.debug:
                msg = r + '[DEBUG]' + n + ' Received ' + g + 'Decline' + n
                msg += ' from ' + bo + src + vendor + ' for '
                msg += ', '.join(addrs) + n
                print(msg)
            return False
        elif p.msgtype == 11:
            if DHCP6OptServerId in p:
                duid = p[DHCP6OptServerId].duid
                if not isinstance(duid, type(self.duid)):
                    return False
                if raw(duid) != raw(self.duid):
                    return False
            if DHCP6OptIA_NA in p or DHCP6OptIA_TA in p or DHCP6OptIA_PD in p:
                return False
        else:
            return False
        return True

    def print_reply(self, req, reply):
        if False:
            while True:
                i = 10

        def norm(s):
            if False:
                return 10
            if s.startswith('DHCPv6 '):
                s = s[7:]
            if s.endswith(' Message'):
                s = s[:-8]
            return s
        if reply is None:
            return
        bo = Color.bold
        g = Color.green + bo
        b = Color.blue + bo
        n = Color.normal
        reqtype = g + norm(req.getlayer(UDP).payload.name) + n
        reqsrc = req.getlayer(IPv6).src
        vendor = in6_addrtovendor(reqsrc)
        if vendor and vendor != 'UNKNOWN':
            vendor = ' [' + b + vendor + n + ']'
        else:
            vendor = ''
        reqsrc = bo + reqsrc + n
        reptype = g + norm(reply.getlayer(UDP).payload.name) + n
        print('Sent %s answering to %s from %s%s' % (reptype, reqtype, reqsrc, vendor))

    def make_reply(self, req):
        if False:
            i = 10
            return i + 15
        p = req[IPv6]
        req_src = p.src
        p = p.payload.payload
        msgtype = p.msgtype
        trid = p.trid

        def _include_options(query, answer):
            if False:
                while True:
                    i = 10
            '\n            Include options from the DHCPv6 query\n            '
            reqopts = []
            if query.haslayer(DHCP6OptOptReq):
                reqopts = query[DHCP6OptOptReq].reqopts
                for (o, opt) in self.dhcpv6_options.items():
                    if o in reqopts:
                        answer /= opt
            else:
                for (o, opt) in self.dhcpv6_options.items():
                    answer /= opt
        if msgtype == 1:
            client_duid = p[DHCP6OptClientId].duid
            resp = IPv6(src=self.src_addr, dst=req_src)
            resp /= UDP(sport=547, dport=546)
            if p.haslayer(DHCP6OptRapidCommit):
                resp /= DHCP6_Reply(trid=trid)
                resp /= DHCP6OptRapidCommit()
                resp /= DHCP6OptServerId(duid=self.duid)
                resp /= DHCP6OptClientId(duid=client_duid)
            elif p.haslayer(DHCP6OptIA_NA) or p.haslayer(DHCP6OptIA_TA):
                msg = 'Scapy6 dhcp6d does not support address assignment'
                resp /= DHCP6_Advertise(trid=trid)
                resp /= DHCP6OptStatusCode(statuscode=2, statusmsg=msg)
                resp /= DHCP6OptServerId(duid=self.duid)
                resp /= DHCP6OptClientId(duid=client_duid)
            elif p.haslayer(DHCP6OptIA_PD):
                msg = 'Scapy6 dhcp6d does not support prefix assignment'
                resp /= DHCP6_Advertise(trid=trid)
                resp /= DHCP6OptStatusCode(statuscode=6, statusmsg=msg)
                resp /= DHCP6OptServerId(duid=self.duid)
                resp /= DHCP6OptClientId(duid=client_duid)
            else:
                resp /= DHCP6_Advertise(trid=trid)
                resp /= DHCP6OptPref(prefval=self.advpref)
                resp /= DHCP6OptServerId(duid=self.duid)
                resp /= DHCP6OptClientId(duid=client_duid)
                resp /= DHCP6OptReconfAccept()
                _include_options(p, resp)
            return resp
        elif msgtype == 3:
            client_duid = p[DHCP6OptClientId].duid
            resp = IPv6(src=self.src_addr, dst=req_src)
            resp /= UDP(sport=547, dport=546)
            resp /= DHCP6_Solicit(trid=trid)
            resp /= DHCP6OptServerId(duid=self.duid)
            resp /= DHCP6OptClientId(duid=client_duid)
            _include_options(p, resp)
            return resp
        elif msgtype == 4:
            pass
        elif msgtype == 5:
            pass
        elif msgtype == 6:
            pass
        elif msgtype == 8:
            pass
        elif msgtype == 9:
            pass
        elif msgtype == 11:
            client_duid = None
            if not p.haslayer(DHCP6OptClientId):
                if self.debug:
                    warning('Received Info Request message without Client Id option')
            else:
                client_duid = p[DHCP6OptClientId].duid
            resp = IPv6(src=self.src_addr, dst=req_src)
            resp /= UDP(sport=547, dport=546)
            resp /= DHCP6_Reply(trid=trid)
            resp /= DHCP6OptServerId(duid=self.duid)
            if client_duid:
                resp /= DHCP6OptClientId(duid=client_duid)
            for (o, opt) in self.dhcpv6_options.items():
                resp /= opt
            return resp
        else:
            pass