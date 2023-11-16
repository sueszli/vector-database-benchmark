"""
Customization for the Solaris operation system.
"""
import socket
from scapy.config import conf
conf.use_pcap = True
socket.IPPROTO_GRE = 47
SIOCGIFHWADDR = 3223349689
from scapy.arch.libpcap import *
from scapy.arch.unix import *
from scapy.interfaces import NetworkInterface

def get_working_if():
    if False:
        for i in range(10):
            print('nop')
    'Return an interface that works'
    try:
        iface = min(conf.route.routes, key=lambda x: x[1])[3]
    except ValueError:
        iface = conf.loopback_name
    return conf.ifaces.dev_from_name(iface)