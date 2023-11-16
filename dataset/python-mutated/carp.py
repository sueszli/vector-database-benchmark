import struct
import hmac
import hashlib
from scapy.packet import Packet, split_layers, bind_layers
from scapy.layers.inet import IP
from scapy.fields import BitField, ByteField, XShortField, XIntField
from scapy.layers.vrrp import IPPROTO_VRRP, VRRP, VRRPv3
from scapy.utils import checksum, inet_aton
from scapy.error import warning

class CARP(Packet):
    name = 'CARP'
    fields_desc = [BitField('version', 4, 4), BitField('type', 4, 4), ByteField('vhid', 1), ByteField('advskew', 0), ByteField('authlen', 0), ByteField('demotion', 0), ByteField('advbase', 0), XShortField('chksum', None), XIntField('counter1', 0), XIntField('counter2', 0), XIntField('hmac1', 0), XIntField('hmac2', 0), XIntField('hmac3', 0), XIntField('hmac4', 0), XIntField('hmac5', 0)]

    def post_build(self, pkt, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.chksum is None:
            pkt = pkt[:6] + struct.pack('!H', checksum(pkt)) + pkt[8:]
        return pkt

    def build_hmac_sha1(self, pw=b'\x00' * 20, ip4l=[], ip6l=[]):
        if False:
            return 10
        h = hmac.new(pw, digestmod=hashlib.sha1)
        h.update(b'!')
        h.update(struct.pack('!B', self.vhid))
        sl = []
        for i in ip4l:
            sl.append(inet_aton(i))
        sl.sort()
        for i in sl:
            h.update(i)
        return h.digest()
warning('CARP overwrites VRRP !')
split_layers(IP, VRRP, proto=IPPROTO_VRRP)
split_layers(IP, VRRPv3, proto=IPPROTO_VRRP)
bind_layers(IP, CARP, proto=112, dst='224.0.0.18')