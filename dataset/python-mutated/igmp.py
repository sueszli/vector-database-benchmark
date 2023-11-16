from scapy.compat import chb, orb
from scapy.error import warning
from scapy.fields import ByteEnumField, ByteField, IPField, XShortField
from scapy.layers.inet import IP, IPOption_Router_Alert
from scapy.layers.l2 import Ether, getmacbyip
from scapy.packet import bind_layers, Packet
from scapy.utils import atol, checksum

def isValidMCAddr(ip):
    if False:
        for i in range(10):
            print('nop')
    'convert dotted quad string to long and check the first octet'
    FirstOct = atol(ip) >> 24 & 255
    return FirstOct >= 224 and FirstOct <= 239

class IGMP(Packet):
    """IGMP Message Class for v1 and v2.

    This class is derived from class Packet. You  need call "igmpize()"
    so the packet is transformed according the RFC when sent.
    a=Ether(src="00:01:02:03:04:05")
    b=IP(src="1.2.3.4")
    c=IGMP(type=0x12, gaddr="224.2.3.4")
    x = a/b/c
    x[IGMP].igmpize()
    sendp(a/b/c, iface="en0")

        Parameters:
          type    IGMP type field, 0x11, 0x12, 0x16 or 0x17
          mrcode  Maximum Response time (zero for v1)
          gaddr   Multicast Group Address 224.x.x.x/4

    See RFC2236, Section 2. Introduction for definitions of proper
    IGMPv2 message format   http://www.faqs.org/rfcs/rfc2236.html
    """
    name = 'IGMP'
    igmptypes = {17: 'Group Membership Query', 18: 'Version 1 - Membership Report', 22: 'Version 2 - Membership Report', 23: 'Leave Group'}
    fields_desc = [ByteEnumField('type', 17, igmptypes), ByteField('mrcode', 20), XShortField('chksum', None), IPField('gaddr', '0.0.0.0')]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        'Called implicitly before a packet is sent to compute and place IGMP checksum.\n\n        Parameters:\n          self    The instantiation of an IGMP class\n          p       The IGMP message in hex in network byte order\n          pay     Additional payload for the IGMP message\n        '
        p += pay
        if self.chksum is None:
            ck = checksum(p)
            p = p[:2] + chb(ck >> 8) + chb(ck & 255) + p[4:]
        return p

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            i = 10
            return i + 15
        if _pkt and len(_pkt) >= 4:
            from scapy.contrib.igmpv3 import IGMPv3
            if orb(_pkt[0]) in [34, 48, 49, 50]:
                return IGMPv3
            if orb(_pkt[0]) == 17 and len(_pkt) >= 12:
                return IGMPv3
        return IGMP

    def igmpize(self):
        if False:
            while True:
                i = 10
        'Called to explicitly fixup the packet according to the IGMP RFC\n\n        The rules are:\n        - General:\n        1.  the Max Response time is meaningful only in Membership Queries and should be zero\n        - IP:\n        1. Send General Group Query to 224.0.0.1 (all systems)\n        2. Send Leave Group to 224.0.0.2 (all routers)\n        3a.Otherwise send the packet to the group address\n        3b.Send reports/joins to the group address\n        4. ttl = 1 (RFC 2236, section 2)\n        5. send the packet with the router alert IP option (RFC 2236, section 2)\n        - Ether:\n        1. Recalculate destination\n\n        Returns:\n            True    The tuple ether/ip/self passed all check and represents\n                    a proper IGMP packet.\n            False   One of more validation checks failed and no fields\n                    were adjusted.\n\n        The function will examine the IGMP message to assure proper format.\n        Corrections will be attempted if possible. The IP header is then properly\n        adjusted to ensure correct formatting and assignment. The Ethernet header\n        is then adjusted to the proper IGMP packet format.\n        '
        from scapy.contrib.igmpv3 import IGMPv3
        gaddr = self.gaddr if hasattr(self, 'gaddr') and self.gaddr else '0.0.0.0'
        underlayer = self.underlayer
        if self.type not in [17, 48]:
            self.mrcode = 0
        if isinstance(underlayer, IP):
            if self.type == 17:
                if gaddr == '0.0.0.0':
                    underlayer.dst = '224.0.0.1'
                elif isValidMCAddr(gaddr):
                    underlayer.dst = gaddr
                else:
                    warning('Invalid IGMP Group Address detected !')
                    return False
            elif self.type == 23 and isValidMCAddr(gaddr):
                underlayer.dst = '224.0.0.2'
            elif (self.type == 18 or self.type == 22) and isValidMCAddr(gaddr):
                underlayer.dst = gaddr
            elif self.type in [17, 34, 48, 49, 50] and isinstance(self, IGMPv3):
                pass
            else:
                warning('Invalid IGMP Type detected !')
                return False
            if not any((isinstance(x, IPOption_Router_Alert) for x in underlayer.options)):
                underlayer.options.append(IPOption_Router_Alert())
            underlayer.ttl = 1
            _root = self.firstlayer()
            if _root.haslayer(Ether):
                _root[Ether].dst = getmacbyip(underlayer.dst)
        if isinstance(self, IGMPv3):
            self.encode_maxrespcode()
        return True

    def mysummary(self):
        if False:
            print('Hello World!')
        'Display a summary of the IGMP object.'
        if isinstance(self.underlayer, IP):
            return self.underlayer.sprintf('IGMP: %IP.src% > %IP.dst% %IGMP.type% %IGMP.gaddr%')
        else:
            return self.sprintf('IGMP %IGMP.type% %IGMP.gaddr%')
bind_layers(IP, IGMP, frag=0, proto=2, ttl=1)