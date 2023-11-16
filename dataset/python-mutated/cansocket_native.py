"""
NativeCANSocket.
"""
import struct
import socket
import time
from scapy.config import conf
from scapy.supersocket import SuperSocket
from scapy.error import Scapy_Exception, warning
from scapy.packet import Packet
from scapy.layers.can import CAN, CAN_MTU, CAN_FD_MTU
from scapy.arch.linux import get_last_packet_timestamp
from scapy.compat import raw
from typing import List, Dict, Type, Any, Optional, Tuple, cast
conf.contribs['NativeCANSocket'] = {'channel': 'can0'}

class NativeCANSocket(SuperSocket):
    """Initializes a Linux PF_CAN socket object.

    Example:
        >>> socket = NativeCANSocket(channel="vcan0", can_filters=[{'can_id': 0x200, 'can_mask': 0x7FF}])

    :param channel: Network interface name
    :param receive_own_messages: Messages, sent by this socket are will
                                 also be received.
    :param can_filters: A list of can filter dictionaries.
    :param basecls: Packet type in which received data gets interpreted.
    :param kwargs: Various keyword arguments for compatibility with
                   PythonCANSockets
    """
    desc = 'read/write packets at a given CAN interface using PF_CAN sockets'

    def __init__(self, channel=None, receive_own_messages=False, can_filters=None, fd=False, basecls=CAN, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        bustype = cast(Optional[str], kwargs.pop('bustype', None))
        if bustype and bustype != 'socketcan':
            warning("You created a NativeCANSocket. If you're providing the argument 'bustype', please use the correct one to achieve compatibility with python-can/PythonCANSocket. \n'bustype=socketcan'")
        self.MTU = CAN_MTU
        self.fd = fd
        self.basecls = basecls
        self.channel = conf.contribs['NativeCANSocket']['channel'] if channel is None else channel
        self.ins = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            self.ins.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, struct.pack('i', receive_own_messages))
        except Exception as exception:
            raise Scapy_Exception('Could not modify receive own messages (%s)', exception)
        if self.fd:
            try:
                self.ins.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
                self.MTU = CAN_FD_MTU
            except Exception as exception:
                raise Scapy_Exception('Could not enable CAN FD support (%s)', exception)
        if can_filters is None:
            can_filters = [{'can_id': 0, 'can_mask': 0}]
        can_filter_fmt = '={}I'.format(2 * len(can_filters))
        filter_data = []
        for can_filter in can_filters:
            filter_data.append(can_filter['can_id'])
            filter_data.append(can_filter['can_mask'])
        self.ins.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, struct.pack(can_filter_fmt, *filter_data))
        self.ins.bind((self.channel,))
        self.outs = self.ins

    def recv_raw(self, x=CAN_MTU):
        if False:
            return 10
        'Returns a tuple containing (cls, pkt_data, time)'
        pkt = None
        try:
            pkt = self.ins.recv(self.MTU)
        except BlockingIOError:
            warning('Captured no data, socket in non-blocking mode.')
        except socket.timeout:
            warning('Captured no data, socket read timed out.')
        except OSError:
            warning('Captured no data.')
        if not conf.contribs['CAN']['swap-bytes'] and pkt is not None:
            pack_fmt = '<I%ds' % (len(pkt) - 4)
            unpack_fmt = '>I%ds' % (len(pkt) - 4)
            pkt = struct.pack(pack_fmt, *struct.unpack(unpack_fmt, pkt))
        return (self.basecls, pkt, get_last_packet_timestamp(self.ins))

    def send(self, x):
        if False:
            return 10
        try:
            x.sent_time = time.time()
        except AttributeError:
            pass
        bs = raw(x)
        if not conf.contribs['CAN']['swap-bytes']:
            pack_fmt = '<I%ds' % (len(bs) - 4)
            unpack_fmt = '>I%ds' % (len(bs) - 4)
            bs = struct.pack(pack_fmt, *struct.unpack(unpack_fmt, bs))
        bs = bs + b'\x00' * (self.MTU - len(bs))
        return super(NativeCANSocket, self).send(bs)
CANSocket = NativeCANSocket