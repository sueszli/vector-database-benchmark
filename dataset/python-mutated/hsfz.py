import logging
import struct
import socket
import time
from scapy.contrib.automotive import log_automotive
from scapy.packet import Packet, bind_layers, bind_bottom_up
from scapy.fields import IntField, ShortEnumField, XByteField
from scapy.layers.inet import TCP
from scapy.supersocket import StreamSocket
from scapy.contrib.automotive.uds import UDS, UDS_TP
from scapy.data import MTU
from typing import Any, Optional, Tuple, Type, Iterable, List, Union
'\nBMW HSFZ (High-Speed-Fahrzeug-Zugang / High-Speed-Car-Access).\nBMW specific diagnostic over IP protocol implementation.\nThe physical interface for this connection is called ENET.\n'

class HSFZ(Packet):
    name = 'HSFZ'
    fields_desc = [IntField('length', None), ShortEnumField('type', 1, {1: 'message', 2: 'echo'}), XByteField('src', 0), XByteField('dst', 0)]

    def hashret(self):
        if False:
            i = 10
            return i + 15
        hdr_hash = struct.pack('B', self.src ^ self.dst)
        pay_hash = self.payload.hashret()
        return hdr_hash + pay_hash

    def extract_padding(self, s):
        if False:
            while True:
                i = 10
        return (s[:self.length - 2], s[self.length - 2:])

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        "\n        This will set the LenField 'length' to the correct value.\n        "
        if self.length is None:
            pkt = struct.pack('!I', len(pay) + 2) + pkt[4:]
        return pkt + pay
bind_bottom_up(TCP, HSFZ, sport=6801)
bind_bottom_up(TCP, HSFZ, dport=6801)
bind_layers(TCP, HSFZ, sport=6801, dport=6801)
bind_layers(HSFZ, UDS)

class HSFZSocket(StreamSocket):

    def __init__(self, ip='127.0.0.1', port=6801):
        if False:
            return 10
        self.ip = ip
        self.port = port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.connect((self.ip, self.port))
        StreamSocket.__init__(self, s, HSFZ)
        self.buffer = b''

    def recv(self, x=MTU, **kwargs):
        if False:
            return 10
        if self.buffer:
            len_data = self.buffer[:4]
        else:
            len_data = self.ins.recv(4, socket.MSG_PEEK)
            if len(len_data) != 4:
                return None
        len_int = struct.unpack('>I', len_data)[0]
        len_int += 6
        self.buffer += self.ins.recv(len_int - len(self.buffer))
        if len(self.buffer) != len_int:
            return None
        pkt = self.basecls(self.buffer, **kwargs)
        self.buffer = b''
        return pkt

class UDS_HSFZSocket(HSFZSocket):

    def __init__(self, src, dst, ip='127.0.0.1', port=6801, basecls=UDS):
        if False:
            i = 10
            return i + 15
        super(UDS_HSFZSocket, self).__init__(ip, port)
        self.src = src
        self.dst = dst
        self.basecls = HSFZ
        self.outputcls = basecls

    def send(self, x):
        if False:
            i = 10
            return i + 15
        try:
            x.sent_time = time.time()
        except AttributeError:
            pass
        try:
            return super(UDS_HSFZSocket, self).send(HSFZ(src=self.src, dst=self.dst) / x)
        except Exception as e:
            log_automotive.exception('Exception: %s', e)
            self.close()
            return 0

    def recv(self, x=MTU, **kwargs):
        if False:
            print('Hello World!')
        pkt = super(UDS_HSFZSocket, self).recv(x)
        if pkt:
            return self.outputcls(bytes(pkt.payload), **kwargs)
        else:
            return pkt

def hsfz_scan(ip, scan_range=range(256), src=244, timeout=0.1, verbose=True):
    if False:
        print('Hello World!')
    '\n    Helper function to scan for HSFZ endpoints.\n\n    Example:\n        >>> sockets = hsfz_scan("192.168.0.42")\n\n    :param ip: IPv4 address of target to scan\n    :param scan_range: Range for HSFZ destination address\n    :param src: HSFZ source address, used during the scan\n    :param timeout: Timeout for each request\n    :param verbose: Show information during scan, if True\n    :return: A list of open UDS_HSFZSockets\n    '
    if verbose:
        log_automotive.setLevel(logging.DEBUG)
    results = list()
    for i in scan_range:
        with UDS_HSFZSocket(src, i, ip) as sock:
            try:
                resp = sock.sr1(UDS() / UDS_TP(), timeout=timeout, verbose=False)
                if resp:
                    results.append((i, resp))
                if resp:
                    log_automotive.debug('Found endpoint %s, src=0x%x, dst=0x%x' % (ip, src, i))
            except Exception as e:
                log_automotive.exception('Error %s at destination address 0x%x' % (e, i))
    return [UDS_HSFZSocket(244, dst, ip) for (dst, _) in results]