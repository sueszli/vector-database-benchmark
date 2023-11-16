"""
PPP (Point to Point Protocol)

[RFC 1661]
"""
import struct
from scapy.config import conf
from scapy.data import DLT_PPP, DLT_PPP_SERIAL, DLT_PPP_ETHER, DLT_PPP_WITH_DIR
from scapy.compat import orb
from scapy.packet import Packet, bind_layers
from scapy.layers.eap import EAP
from scapy.layers.l2 import Ether, CookedLinux, GRE_PPTP
from scapy.layers.inet import IP
from scapy.layers.inet6 import IPv6
from scapy.fields import BitField, ByteEnumField, ByteField, ConditionalField, EnumField, FieldLenField, IPField, IntField, OUIField, PacketField, PacketListField, ShortEnumField, ShortField, StrLenField, XByteField, XShortField, XStrLenField

class PPPoE(Packet):
    name = 'PPP over Ethernet'
    fields_desc = [BitField('version', 1, 4), BitField('type', 1, 4), ByteEnumField('code', 0, {0: 'Session'}), XShortField('sessionid', 0), ShortField('len', None)]

    def post_build(self, p, pay):
        if False:
            return 10
        p += pay
        if self.len is None:
            tmp_len = len(p) - 6
            p = p[:4] + struct.pack('!H', tmp_len) + p[6:]
        return p

class PPPoED(PPPoE):
    name = 'PPP over Ethernet Discovery'
    code_list = {0: 'PPP Session Stage', 9: 'PPPoE Active Discovery Initiation (PADI)', 7: 'PPPoE Active Discovery Offer (PADO)', 10: 'PPPoE Active Discovery Session-Grant (PADG)', 11: 'PPPoE Active Discovery Session-Credit Response (PADC)', 12: 'PPPoE Active Discovery Quality (PADQ)', 25: 'PPPoE Active Discovery Request (PADR)', 101: 'PPPoE Active Discovery Session-confirmation (PADS)', 167: 'PPPoE Active Discovery Terminate (PADT)'}
    fields_desc = [BitField('version', 1, 4), BitField('type', 1, 4), ByteEnumField('code', 9, code_list), XShortField('sessionid', 0), ShortField('len', None)]

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        return (s[:self.len], s[self.len:])

    def mysummary(self):
        if False:
            return 10
        return self.sprintf('%code%')

class PPPoETag(Packet):
    name = 'PPPoE Tag'
    tag_list = {0: 'End-Of-List', 257: 'Service-Name', 258: 'AC-Name', 259: 'Host-Uniq', 260: 'AC-Cookie', 261: 'Vendor-Specific', 262: 'Credits', 263: 'Metrics', 264: 'Sequence Number', 265: 'Credit Scale Factor', 272: 'Relay-Session-Id', 288: 'PPP-Max-Payload', 513: 'Service-Name-Error', 514: 'AC-System-Error', 515: 'Generic-Error'}
    fields_desc = [ShortEnumField('tag_type', None, tag_list), FieldLenField('tag_len', None, length_of='tag_value', fmt='H'), StrLenField('tag_value', '', length_from=lambda pkt: pkt.tag_len)]

    def extract_padding(self, s):
        if False:
            while True:
                i = 10
        return ('', s)

class PPPoED_Tags(Packet):
    name = 'PPPoE Tag List'
    fields_desc = [PacketListField('tag_list', None, PPPoETag)]

    def mysummary(self):
        if False:
            while True:
                i = 10
        return ('PPPoE Tags' + ', '.join((x.sprintf('%tag_type%') for x in self.tag_list)), [PPPoED])
_PPP_PROTOCOLS = {1: 'Padding Protocol', 3: 'ROHC small-CID [RFC3095]', 5: 'ROHC large-CID [RFC3095]', 33: 'Internet Protocol version 4', 35: 'OSI Network Layer', 37: 'Xerox NS IDP', 39: 'DECnet Phase IV', 41: 'Appletalk', 43: 'Novell IPX', 45: 'Van Jacobson Compressed TCP/IP', 47: 'Van Jacobson Uncompressed TCP/IP', 49: 'Bridging PDU', 51: 'Stream Protocol (ST-II)', 53: 'Banyan Vines', 55: 'reserved (until 1993) [Typo in RFC1172]', 57: 'AppleTalk EDDP', 59: 'AppleTalk SmartBuffered', 61: 'Multi-Link [RFC1717]', 63: 'NETBIOS Framing', 65: 'Cisco Systems', 67: 'Ascom Timeplex', 69: 'Fujitsu Link Backup and Load Balancing (LBLB)', 71: 'DCA Remote Lan', 73: 'Serial Data Transport Protocol (PPP-SDTP)', 75: 'SNA over 802.2', 77: 'SNA', 79: 'IPv6 Header Compression', 81: 'KNX Bridging Data [ianp]', 83: 'Encryption [Meyer]', 85: 'Individual Link Encryption [Meyer]', 87: 'Internet Protocol version 6 [Hinden]', 89: 'PPP Muxing [RFC3153]', 91: 'Vendor-Specific Network Protocol (VSNP) [RFC3772]', 97: 'RTP IPHC Full Header [RFC3544]', 99: 'RTP IPHC Compressed TCP [RFC3544]', 101: 'RTP IPHC Compressed Non TCP [RFC3544]', 103: 'RTP IPHC Compressed UDP 8 [RFC3544]', 105: 'RTP IPHC Compressed RTP 8 [RFC3544]', 111: 'Stampede Bridging', 113: 'Reserved [Fox]', 115: 'MP+ Protocol [Smith]', 125: 'reserved (Control Escape) [RFC1661]', 127: 'reserved (compression inefficient [RFC1662]', 129: 'Reserved Until 20-Oct-2000 [IANA]', 131: 'Reserved Until 20-Oct-2000 [IANA]', 193: 'NTCITS IPI [Ungar]', 207: 'reserved (PPP NLID)', 251: 'single link compression in multilink [RFC1962]', 253: 'compressed datagram [RFC1962]', 255: 'reserved (compression inefficient)', 513: '802.1d Hello Packets', 515: 'IBM Source Routing BPDU', 517: 'DEC LANBridge100 Spanning Tree', 519: 'Cisco Discovery Protocol [Sastry]', 521: 'Netcs Twin Routing [Korfmacher]', 523: 'STP - Scheduled Transfer Protocol [Segal]', 525: 'EDP - Extreme Discovery Protocol [Grosser]', 529: 'Optical Supervisory Channel Protocol (OSCP)[Prasad]', 531: 'Optical Supervisory Channel Protocol (OSCP)[Prasad]', 561: 'Luxcom', 563: 'Sigma Network Systems', 565: 'Apple Client Server Protocol [Ridenour]', 641: 'MPLS Unicast [RFC3032]  ', 643: 'MPLS Multicast [RFC3032]', 645: 'IEEE p1284.4 standard - data packets [Batchelder]', 647: 'ETSI TETRA Network Protocol Type 1 [Nieminen]', 649: 'Multichannel Flow Treatment Protocol [McCann]', 8291: 'RTP IPHC Compressed TCP No Delta [RFC3544]', 8293: 'RTP IPHC Context State [RFC3544]', 8295: 'RTP IPHC Compressed UDP 16 [RFC3544]', 8297: 'RTP IPHC Compressed RTP 16 [RFC3544]', 16385: 'Cray Communications Control Protocol [Stage]', 16387: 'CDPD Mobile Network Registration Protocol [Quick]', 16389: 'Expand accelerator protocol [Rachmani]', 16391: 'ODSICP NCP [Arvind]', 16393: 'DOCSIS DLL [Gaedtke]', 16395: 'Cetacean Network Detection Protocol [Siller]', 16417: 'Stacker LZS [Simpson]', 16419: 'RefTek Protocol [Banfill]', 16421: 'Fibre Channel [Rajagopal]', 16423: 'EMIT Protocols [Eastham]', 16475: 'Vendor-Specific Protocol (VSP) [RFC3772]', 32801: 'Internet Protocol Control Protocol', 32803: 'OSI Network Layer Control Protocol', 32805: 'Xerox NS IDP Control Protocol', 32807: 'DECnet Phase IV Control Protocol', 32809: 'Appletalk Control Protocol', 32811: 'Novell IPX Control Protocol', 32813: 'reserved', 32815: 'reserved', 32817: 'Bridging NCP', 32819: 'Stream Protocol Control Protocol', 32821: 'Banyan Vines Control Protocol', 32823: 'reserved (until 1993)', 32825: 'reserved', 32827: 'reserved', 32829: 'Multi-Link Control Protocol', 32831: 'NETBIOS Framing Control Protocol', 32833: 'Cisco Systems Control Protocol', 32835: 'Ascom Timeplex', 32837: 'Fujitsu LBLB Control Protocol', 32839: 'DCA Remote Lan Network Control Protocol (RLNCP)', 32841: 'Serial Data Control Protocol (PPP-SDCP)', 32843: 'SNA over 802.2 Control Protocol', 32845: 'SNA Control Protocol', 32847: 'IP6 Header Compression Control Protocol', 32849: 'KNX Bridging Control Protocol [ianp]', 32851: 'Encryption Control Protocol [Meyer]', 32853: 'Individual Link Encryption Control Protocol [Meyer]', 32855: 'IPv6 Control Protovol [Hinden]', 32857: 'PPP Muxing Control Protocol [RFC3153]', 32859: 'Vendor-Specific Network Control Protocol (VSNCP) [RFC3772]', 32879: 'Stampede Bridging Control Protocol', 32883: 'MP+ Control Protocol [Smith]', 32881: 'Reserved [Fox]', 32893: 'Not Used - reserved [RFC1661]', 32897: 'Reserved Until 20-Oct-2000 [IANA]', 32899: 'Reserved Until 20-Oct-2000 [IANA]', 32961: 'NTCITS IPI Control Protocol [Ungar]', 32975: 'Not Used - reserved [RFC1661]', 33019: 'single link compression in multilink control [RFC1962]', 33021: 'Compression Control Protocol [RFC1962]', 33023: 'Not Used - reserved [RFC1661]', 33287: 'Cisco Discovery Protocol Control [Sastry]', 33289: 'Netcs Twin Routing [Korfmacher]', 33291: 'STP - Control Protocol [Segal]', 33293: 'EDPCP - Extreme Discovery Protocol Ctrl Prtcl [Grosser]', 33333: 'Apple Client Server Protocol Control [Ridenour]', 33409: 'MPLSCP [RFC3032]', 33413: 'IEEE p1284.4 standard - Protocol Control [Batchelder]', 33415: 'ETSI TETRA TNP1 Control Protocol [Nieminen]', 33417: 'Multichannel Flow Treatment Protocol [McCann]', 49185: 'Link Control Protocol', 49187: 'Password Authentication Protocol', 49189: 'Link Quality Report', 49191: 'Shiva Password Authentication Protocol', 49193: 'CallBack Control Protocol (CBCP)', 49195: 'BACP Bandwidth Allocation Control Protocol [RFC2125]', 49197: 'BAP [RFC2125]', 49243: 'Vendor-Specific Authentication Protocol (VSAP) [RFC3772]', 49281: 'Container Control Protocol [KEN]', 49699: 'Challenge Handshake Authentication Protocol', 49701: 'RSA Authentication Protocol [Narayana]', 49703: 'Extensible Authentication Protocol [RFC2284]', 49705: 'Mitsubishi Security Info Exch Ptcl (SIEP) [Seno]', 49775: 'Stampede Bridging Authorization Protocol', 49793: 'Proprietary Authentication Protocol [KEN]', 49795: 'Proprietary Authentication Protocol [Tackabury]', 50305: 'Proprietary Node ID Authentication Protocol [KEN]'}

class HDLC(Packet):
    fields_desc = [XByteField('address', 255), XByteField('control', 3)]

class DIR_PPP(Packet):
    fields_desc = [ByteEnumField('direction', 0, ['received', 'sent'])]

class _PPPProtoField(EnumField):
    """
    A field that can be either Byte or Short, depending on the PPP RFC.

    See RFC 1661 section 2
    <https://tools.ietf.org/html/rfc1661#section-2>

    The generated proto field is two bytes when not specified, or when specified
    as an integer or a string:
      PPP()
      PPP(proto=0x21)
      PPP(proto="Internet Protocol version 4")
    To explicitly forge a one byte proto field, use the bytes representation:
      PPP(proto=b'!')
    """

    def getfield(self, pkt, s):
        if False:
            while True:
                i = 10
        if ord(s[:1]) & 1:
            self.fmt = '!B'
            self.sz = 1
        else:
            self.fmt = '!H'
            self.sz = 2
        self.struct = struct.Struct(self.fmt)
        return super(_PPPProtoField, self).getfield(pkt, s)

    def addfield(self, pkt, s, val):
        if False:
            i = 10
            return i + 15
        if isinstance(val, bytes):
            if len(val) == 1:
                (fmt, sz) = ('!B', 1)
            elif len(val) == 2:
                (fmt, sz) = ('!H', 2)
            else:
                raise TypeError('Invalid length for PPP proto')
            val = struct.Struct(fmt).unpack(val)[0]
        else:
            (fmt, sz) = ('!H', 2)
        self.fmt = fmt
        self.sz = sz
        self.struct = struct.Struct(self.fmt)
        return super(_PPPProtoField, self).addfield(pkt, s, val)

class PPP(Packet):
    name = 'PPP Link Layer'
    fields_desc = [_PPPProtoField('proto', 33, _PPP_PROTOCOLS)]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            return 10
        if _pkt and _pkt[:1] == b'\xff':
            return HDLC
        return cls
_PPP_conftypes = {1: 'Configure-Request', 2: 'Configure-Ack', 3: 'Configure-Nak', 4: 'Configure-Reject', 5: 'Terminate-Request', 6: 'Terminate-Ack', 7: 'Code-Reject', 8: 'Protocol-Reject', 9: 'Echo-Request', 10: 'Echo-Reply', 11: 'Discard-Request', 14: 'Reset-Request', 15: 'Reset-Ack'}
_PPP_ipcpopttypes = {1: 'IP-Addresses (Deprecated)', 2: 'IP-Compression-Protocol', 3: 'IP-Address', 4: 'Mobile-IPv4', 129: 'Primary-DNS-Address', 130: 'Primary-NBNS-Address', 131: 'Secondary-DNS-Address', 132: 'Secondary-NBNS-Address'}

class PPP_IPCP_Option(Packet):
    name = 'PPP IPCP Option'
    fields_desc = [ByteEnumField('type', None, _PPP_ipcpopttypes), FieldLenField('len', None, length_of='data', fmt='B', adjust=lambda _, val: val + 2), StrLenField('data', '', length_from=lambda pkt: max(0, pkt.len - 2))]

    def extract_padding(self, pay):
        if False:
            i = 10
            return i + 15
        return (b'', pay)
    registered_options = {}

    @classmethod
    def register_variant(cls):
        if False:
            return 10
        cls.registered_options[cls.type.default] = cls

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            print('Hello World!')
        if _pkt:
            o = orb(_pkt[0])
            return cls.registered_options.get(o, cls)
        return cls

class PPP_IPCP_Option_IPAddress(PPP_IPCP_Option):
    name = 'PPP IPCP Option: IP Address'
    fields_desc = [ByteEnumField('type', 3, _PPP_ipcpopttypes), FieldLenField('len', None, length_of='data', fmt='B', adjust=lambda _, val: val + 2), IPField('data', '0.0.0.0'), StrLenField('garbage', '', length_from=lambda pkt: pkt.len - 6)]

class PPP_IPCP_Option_DNS1(PPP_IPCP_Option_IPAddress):
    name = 'PPP IPCP Option: DNS1 Address'
    type = 129

class PPP_IPCP_Option_DNS2(PPP_IPCP_Option_IPAddress):
    name = 'PPP IPCP Option: DNS2 Address'
    type = 131

class PPP_IPCP_Option_NBNS1(PPP_IPCP_Option_IPAddress):
    name = 'PPP IPCP Option: NBNS1 Address'
    type = 130

class PPP_IPCP_Option_NBNS2(PPP_IPCP_Option_IPAddress):
    name = 'PPP IPCP Option: NBNS2 Address'
    type = 132

class PPP_IPCP(Packet):
    fields_desc = [ByteEnumField('code', 1, _PPP_conftypes), XByteField('id', 0), FieldLenField('len', None, fmt='H', length_of='options', adjust=lambda _, val: val + 4), PacketListField('options', [], PPP_IPCP_Option, length_from=lambda pkt: pkt.len - 4)]
_PPP_ecpopttypes = {0: 'OUI', 1: 'DESE'}

class PPP_ECP_Option(Packet):
    name = 'PPP ECP Option'
    fields_desc = [ByteEnumField('type', None, _PPP_ecpopttypes), FieldLenField('len', None, length_of='data', fmt='B', adjust=lambda _, val: val + 2), StrLenField('data', '', length_from=lambda pkt: max(0, pkt.len - 2))]

    def extract_padding(self, pay):
        if False:
            while True:
                i = 10
        return (b'', pay)
    registered_options = {}

    @classmethod
    def register_variant(cls):
        if False:
            return 10
        cls.registered_options[cls.type.default] = cls

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            return 10
        if _pkt:
            o = orb(_pkt[0])
            return cls.registered_options.get(o, cls)
        return cls

class PPP_ECP_Option_OUI(PPP_ECP_Option):
    fields_desc = [ByteEnumField('type', 0, _PPP_ecpopttypes), FieldLenField('len', None, length_of='data', fmt='B', adjust=lambda _, val: val + 6), OUIField('oui', 0), ByteField('subtype', 0), StrLenField('data', '', length_from=lambda pkt: pkt.len - 6)]

class PPP_ECP(Packet):
    fields_desc = [ByteEnumField('code', 1, _PPP_conftypes), XByteField('id', 0), FieldLenField('len', None, fmt='H', length_of='options', adjust=lambda _, val: val + 4), PacketListField('options', [], PPP_ECP_Option, length_from=lambda pkt: pkt.len - 4)]
_PPP_lcptypes = {1: 'Configure-Request', 2: 'Configure-Ack', 3: 'Configure-Nak', 4: 'Configure-Reject', 5: 'Terminate-Request', 6: 'Terminate-Ack', 7: 'Code-Reject', 8: 'Protocol-Reject', 9: 'Echo-Request', 10: 'Echo-Reply', 11: 'Discard-Request'}

class PPP_LCP(Packet):
    name = 'PPP Link Control Protocol'
    fields_desc = [ByteEnumField('code', 5, _PPP_lcptypes), XByteField('id', 0), FieldLenField('len', None, fmt='H', length_of='data', adjust=lambda _, val: val + 4), StrLenField('data', '', length_from=lambda pkt: pkt.len - 4)]

    def mysummary(self):
        if False:
            while True:
                i = 10
        return self.sprintf('LCP %code%')

    def extract_padding(self, pay):
        if False:
            for i in range(10):
                print('nop')
        return (b'', pay)

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            return 10
        if _pkt:
            o = orb(_pkt[0])
            if o in [1, 2, 3, 4]:
                return PPP_LCP_Configure
            elif o in [5, 6]:
                return PPP_LCP_Terminate
            elif o == 7:
                return PPP_LCP_Code_Reject
            elif o == 8:
                return PPP_LCP_Protocol_Reject
            elif o in [9, 10]:
                return PPP_LCP_Echo
            elif o == 11:
                return PPP_LCP_Discard_Request
            else:
                return cls
        return cls
_PPP_lcp_optiontypes = {1: 'Maximum-Receive-Unit', 2: 'Async-Control-Character-Map', 3: 'Authentication-protocol', 4: 'Quality-protocol', 5: 'Magic-number', 7: 'Protocol-Field-Compression', 8: 'Address-and-Control-Field-Compression', 13: 'Callback'}

class PPP_LCP_Option(Packet):
    name = 'PPP LCP Option'
    fields_desc = [ByteEnumField('type', None, _PPP_lcp_optiontypes), FieldLenField('len', None, fmt='B', length_of='data', adjust=lambda _, val: val + 2), StrLenField('data', None, length_from=lambda pkt: pkt.len - 2)]

    def extract_padding(self, pay):
        if False:
            for i in range(10):
                print('nop')
        return (b'', pay)
    registered_options = {}

    @classmethod
    def register_variant(cls):
        if False:
            while True:
                i = 10
        cls.registered_options[cls.type.default] = cls

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        if _pkt:
            o = orb(_pkt[0])
            return cls.registered_options.get(o, cls)
        return cls

class PPP_LCP_MRU_Option(PPP_LCP_Option):
    fields_desc = [ByteEnumField('type', 1, _PPP_lcp_optiontypes), ByteField('len', 4), ShortField('max_recv_unit', 1500)]
_PPP_LCP_auth_protocols = {49187: 'Password authentication protocol', 49699: 'Challenge-response authentication protocol', 49703: 'PPP Extensible authentication protocol'}
_PPP_LCP_CHAP_algorithms = {5: 'MD5', 6: 'SHA1', 128: 'MS-CHAP', 129: 'MS-CHAP-v2'}

class PPP_LCP_ACCM_Option(PPP_LCP_Option):
    fields_desc = [ByteEnumField('type', 2, _PPP_lcp_optiontypes), ByteField('len', 6), BitField('accm', 0, 32)]

def adjust_auth_len(pkt, x):
    if False:
        i = 10
        return i + 15
    if pkt.auth_protocol == 49699:
        return 5
    elif pkt.auth_protocol == 49187:
        return 4
    else:
        return x + 4

class PPP_LCP_Auth_Protocol_Option(PPP_LCP_Option):
    fields_desc = [ByteEnumField('type', 3, _PPP_lcp_optiontypes), FieldLenField('len', None, fmt='B', length_of='data', adjust=adjust_auth_len), ShortEnumField('auth_protocol', 49187, _PPP_LCP_auth_protocols), ConditionalField(StrLenField('data', '', length_from=lambda pkt: pkt.len - 4), lambda pkt: pkt.auth_protocol != 49699), ConditionalField(ByteEnumField('algorithm', 5, _PPP_LCP_CHAP_algorithms), lambda pkt: pkt.auth_protocol == 49699)]
_PPP_LCP_quality_protocols = {49189: 'Link Quality Report'}

class PPP_LCP_Quality_Protocol_Option(PPP_LCP_Option):
    fields_desc = [ByteEnumField('type', 4, _PPP_lcp_optiontypes), FieldLenField('len', None, fmt='B', length_of='data', adjust=lambda _, val: val + 4), ShortEnumField('quality_protocol', 49189, _PPP_LCP_quality_protocols), StrLenField('data', '', length_from=lambda pkt: pkt.len - 4)]

class PPP_LCP_Magic_Number_Option(PPP_LCP_Option):
    fields_desc = [ByteEnumField('type', 5, _PPP_lcp_optiontypes), ByteField('len', 6), IntField('magic_number', None)]
_PPP_lcp_callback_operations = {0: 'Location determined by user authentication', 1: 'Dialing string', 2: 'Location identifier', 3: 'E.164 number', 4: 'Distinguished name'}

class PPP_LCP_Callback_Option(PPP_LCP_Option):
    fields_desc = [ByteEnumField('type', 13, _PPP_lcp_optiontypes), FieldLenField('len', None, fmt='B', length_of='message', adjust=lambda _, val: val + 3), ByteEnumField('operation', 0, _PPP_lcp_callback_operations), StrLenField('message', '', length_from=lambda pkt: pkt.len - 3)]

class PPP_LCP_Configure(PPP_LCP):
    fields_desc = [ByteEnumField('code', 1, _PPP_lcptypes), XByteField('id', 0), FieldLenField('len', None, fmt='H', length_of='options', adjust=lambda _, val: val + 4), PacketListField('options', [], PPP_LCP_Option, length_from=lambda pkt: pkt.len - 4)]

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, PPP_LCP_Configure) and self.code in [2, 3, 4] and (other.code == 1) and (other.id == self.id)

class PPP_LCP_Terminate(PPP_LCP):

    def answers(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, PPP_LCP_Terminate) and self.code == 6 and (other.code == 5) and (other.id == self.id)

class PPP_LCP_Code_Reject(PPP_LCP):
    fields_desc = [ByteEnumField('code', 7, _PPP_lcptypes), XByteField('id', 0), FieldLenField('len', None, fmt='H', length_of='rejected_packet', adjust=lambda _, val: val + 4), PacketField('rejected_packet', None, PPP_LCP)]

class PPP_LCP_Protocol_Reject(PPP_LCP):
    fields_desc = [ByteEnumField('code', 8, _PPP_lcptypes), XByteField('id', 0), FieldLenField('len', None, fmt='H', length_of='rejected_information', adjust=lambda _, val: val + 6), ShortEnumField('rejected_protocol', None, _PPP_PROTOCOLS), PacketField('rejected_information', None, Packet)]

class PPP_LCP_Discard_Request(PPP_LCP):
    fields_desc = [ByteEnumField('code', 11, _PPP_lcptypes), XByteField('id', 0), FieldLenField('len', None, fmt='H', length_of='data', adjust=lambda _, val: val + 8), IntField('magic_number', None), StrLenField('data', '', length_from=lambda pkt: pkt.len - 8)]

class PPP_LCP_Echo(PPP_LCP_Discard_Request):
    code = 9

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, PPP_LCP_Echo) and self.code == 10 and (other.code == 9) and (self.id == other.id)
_PPP_paptypes = {1: 'Authenticate-Request', 2: 'Authenticate-Ack', 3: 'Authenticate-Nak'}

class PPP_PAP(Packet):
    name = 'PPP Password Authentication Protocol'
    fields_desc = [ByteEnumField('code', 1, _PPP_paptypes), XByteField('id', 0), FieldLenField('len', None, fmt='!H', length_of='data', adjust=lambda _, val: val + 4), StrLenField('data', '', length_from=lambda pkt: pkt.len - 4)]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *_, **kargs):
        if False:
            print('Hello World!')
        code = None
        if _pkt:
            code = orb(_pkt[0])
        elif 'code' in kargs:
            code = kargs['code']
            if isinstance(code, str):
                code = cls.fields_desc[0].s2i[code]
        if code == 1:
            return PPP_PAP_Request
        elif code in [2, 3]:
            return PPP_PAP_Response
        return cls

    def extract_padding(self, pay):
        if False:
            return 10
        return ('', pay)

class PPP_PAP_Request(PPP_PAP):
    fields_desc = [ByteEnumField('code', 1, _PPP_paptypes), XByteField('id', 0), FieldLenField('len', None, fmt='!H', length_of='username', adjust=lambda pkt, val: val + 6 + len(pkt.password)), FieldLenField('username_len', None, fmt='B', length_of='username'), StrLenField('username', None, length_from=lambda pkt: pkt.username_len), FieldLenField('passwd_len', None, fmt='B', length_of='password'), StrLenField('password', None, length_from=lambda pkt: pkt.passwd_len)]

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        return self.sprintf('PAP-Request username=%PPP_PAP_Request.username% password=%PPP_PAP_Request.password%')

class PPP_PAP_Response(PPP_PAP):
    fields_desc = [ByteEnumField('code', 2, _PPP_paptypes), XByteField('id', 0), FieldLenField('len', None, fmt='!H', length_of='message', adjust=lambda _, val: val + 5), FieldLenField('msg_len', None, fmt='B', length_of='message'), StrLenField('message', '', length_from=lambda pkt: pkt.msg_len)]

    def answers(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, PPP_PAP_Request) and other.id == self.id

    def mysummary(self):
        if False:
            for i in range(10):
                print('nop')
        res = 'PAP-Ack' if self.code == 2 else 'PAP-Nak'
        if self.msg_len > 0:
            res += self.sprintf(' msg=%PPP_PAP_Response.message%')
        return res
_PPP_chaptypes = {1: 'Challenge', 2: 'Response', 3: 'Success', 4: 'Failure'}

class PPP_CHAP(Packet):
    name = 'PPP Challenge Handshake Authentication Protocol'
    fields_desc = [ByteEnumField('code', 1, _PPP_chaptypes), XByteField('id', 0), FieldLenField('len', None, fmt='!H', length_of='data', adjust=lambda _, val: val + 4), StrLenField('data', '', length_from=lambda pkt: pkt.len - 4)]

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, PPP_CHAP_ChallengeResponse) and other.code == 2 and (self.code in (3, 4)) and (self.id == other.id)

    @classmethod
    def dispatch_hook(cls, _pkt=None, *_, **kargs):
        if False:
            i = 10
            return i + 15
        code = None
        if _pkt:
            code = orb(_pkt[0])
        elif 'code' in kargs:
            code = kargs['code']
            if isinstance(code, str):
                code = cls.fields_desc[0].s2i[code]
        if code in (1, 2):
            return PPP_CHAP_ChallengeResponse
        return cls

    def extract_padding(self, pay):
        if False:
            while True:
                i = 10
        return ('', pay)

    def mysummary(self):
        if False:
            return 10
        if self.code == 3:
            return self.sprintf('CHAP Success message=%PPP_CHAP.data%')
        elif self.code == 4:
            return self.sprintf('CHAP Failure message=%PPP_CHAP.data%')

class PPP_CHAP_ChallengeResponse(PPP_CHAP):
    fields_desc = [ByteEnumField('code', 1, _PPP_chaptypes), XByteField('id', 0), FieldLenField('len', None, fmt='!H', length_of='value', adjust=lambda pkt, val: val + len(pkt.optional_name) + 5), FieldLenField('value_size', None, fmt='B', length_of='value'), XStrLenField('value', b'\x00\x00\x00\x00\x00\x00\x00\x00', length_from=lambda pkt: pkt.value_size), StrLenField('optional_name', '', length_from=lambda pkt: pkt.len - pkt.value_size - 5)]

    def answers(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, PPP_CHAP_ChallengeResponse) and other.code == 1 and (self.code == 2) and (self.id == other.id)

    def mysummary(self):
        if False:
            i = 10
            return i + 15
        if self.code == 1:
            return self.sprintf('CHAP challenge=0x%PPP_CHAP_ChallengeResponse.value% optional_name=%PPP_CHAP_ChallengeResponse.optional_name%')
        elif self.code == 2:
            return self.sprintf('CHAP response=0x%PPP_CHAP_ChallengeResponse.value% optional_name=%PPP_CHAP_ChallengeResponse.optional_name%')
        else:
            return super(PPP_CHAP_ChallengeResponse, self).mysummary()
bind_layers(PPPoED, PPPoED_Tags, type=1)
bind_layers(Ether, PPPoED, type=34915)
bind_layers(Ether, PPPoE, type=34916)
bind_layers(CookedLinux, PPPoED, proto=34915)
bind_layers(CookedLinux, PPPoE, proto=34916)
bind_layers(PPPoE, PPP, code=0)
bind_layers(HDLC, PPP)
bind_layers(DIR_PPP, PPP)
bind_layers(PPP, EAP, proto=49703)
bind_layers(PPP, IP, proto=33)
bind_layers(PPP, IPv6, proto=87)
bind_layers(PPP, PPP_CHAP, proto=49699)
bind_layers(PPP, PPP_IPCP, proto=32801)
bind_layers(PPP, PPP_ECP, proto=32851)
bind_layers(PPP, PPP_LCP, proto=49185)
bind_layers(PPP, PPP_PAP, proto=49187)
bind_layers(Ether, PPP_IPCP, type=32801)
bind_layers(Ether, PPP_ECP, type=32851)
bind_layers(GRE_PPTP, PPP, proto=34827)
conf.l2types.register(DLT_PPP, PPP)
conf.l2types.register(DLT_PPP_SERIAL, HDLC)
conf.l2types.register(DLT_PPP_ETHER, PPPoE)
conf.l2types.register(DLT_PPP_WITH_DIR, DIR_PPP)