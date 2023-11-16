"""
Python-CAN CANSocket Wrapper.
"""
import time
import struct
import threading
from functools import reduce
from operator import add
from collections import deque
from scapy.config import conf
from scapy.supersocket import SuperSocket
from scapy.layers.can import CAN
from scapy.packet import Packet
from scapy.error import warning
from typing import List, Type, Tuple, Dict, Any, Optional, cast
from can import Message as can_Message
from can import CanError as can_CanError
from can import BusABC as can_BusABC
from can.interface import Bus as can_Bus
__all__ = ['CANSocket', 'PythonCANSocket']

class SocketMapper(object):
    """Internal Helper class to map a python-can bus object to
    a list of SocketWrapper instances
    """

    def __init__(self, bus, sockets):
        if False:
            return 10
        'Initializes the SocketMapper helper class\n\n        :param bus: A python-can Bus object\n        :param sockets: A list of SocketWrapper objects which want to receive\n                        messages from the provided python-can Bus object.\n        '
        self.bus = bus
        self.sockets = sockets

    def mux(self):
        if False:
            print('Hello World!')
        'Multiplexer function. Tries to receive from its python-can bus\n        object. If a message is received, this message gets forwarded to\n        all receive queues of the SocketWrapper objects.\n        '
        msgs = []
        while True:
            try:
                msg = self.bus.recv(timeout=0)
                if msg is None:
                    break
                else:
                    msgs.append(msg)
            except Exception as e:
                warning('[MUX] python-can exception caught: %s' % e)
        for sock in self.sockets:
            with sock.lock:
                for msg in msgs:
                    if sock._matches_filters(msg):
                        sock.rx_queue.append(msg)

class _SocketsPool(object):
    """Helper class to organize all SocketWrapper and SocketMapper objects"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.pool = dict()
        self.pool_mutex = threading.Lock()
        self.last_call = 0.0

    def internal_send(self, sender, msg):
        if False:
            for i in range(10):
                print('nop')
        'Internal send function.\n\n        A given SocketWrapper wants to send a CAN message. The python-can\n        Bus object is obtained from an internal pool of SocketMapper objects.\n        The given message is sent on the python-can Bus object and also\n        inserted into the message queues of all other SocketWrapper objects\n        which are connected to the same python-can bus object\n        by the SocketMapper.\n\n        :param sender: SocketWrapper which initiated a send of a CAN message\n        :param msg: CAN message to be sent\n        '
        if sender.name is None:
            raise TypeError('SocketWrapper.name should never be None')
        with self.pool_mutex:
            try:
                mapper = self.pool[sender.name]
                mapper.bus.send(msg)
                for sock in mapper.sockets:
                    if sock == sender:
                        continue
                    if not sock._matches_filters(msg):
                        continue
                    with sock.lock:
                        sock.rx_queue.append(msg)
            except KeyError:
                warning('[SND] Socket %s not found in pool' % sender.name)
            except can_CanError as e:
                warning('[SND] python-can exception caught: %s' % e)

    def multiplex_rx_packets(self):
        if False:
            for i in range(10):
                print('nop')
        'This calls the mux() function of all SocketMapper\n        objects in this SocketPool\n        '
        if time.monotonic() - self.last_call < 0.001:
            return
        with self.pool_mutex:
            for t in self.pool.values():
                t.mux()
        self.last_call = time.monotonic()

    def register(self, socket, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Registers a SocketWrapper object. Every SocketWrapper describes to\n        a python-can bus object. This python-can bus object can only exist\n        once. In case this object already exists in this SocketsPool, organized\n        by a SocketMapper object, the new SocketWrapper is inserted in the\n        list of subscribers of the SocketMapper. Otherwise a new python-can\n        Bus object is created from the provided args and kwargs and inserted,\n        encapsulated in a SocketMapper, into this SocketsPool.\n\n        :param socket: SocketWrapper object which needs to be registered.\n        :param args: Arguments for the python-can Bus object\n        :param kwargs: Keyword arguments for the python-can Bus object\n        '
        if 'interface' in kwargs.keys():
            k = str(kwargs.get('interface', 'unknown_interface')) + '_' + str(kwargs.get('channel', 'unknown_channel'))
        else:
            k = str(kwargs.get('bustype', 'unknown_bustype')) + '_' + str(kwargs.get('channel', 'unknown_channel'))
        with self.pool_mutex:
            if k in self.pool:
                t = self.pool[k]
                t.sockets.append(socket)
                filters = [s.filters for s in t.sockets if s.filters is not None]
                if filters:
                    t.bus.set_filters(reduce(add, filters))
                socket.name = k
            else:
                bus = can_Bus(*args, **kwargs)
                socket.name = k
                self.pool[k] = SocketMapper(bus, [socket])

    def unregister(self, socket):
        if False:
            return 10
        "Unregisters a SocketWrapper from its subscription to a SocketMapper.\n\n        If a SocketMapper doesn't have any subscribers, the python-can Bus\n        get shutdown.\n\n        :param socket: SocketWrapper to be unregistered\n        "
        if socket.name is None:
            raise TypeError('SocketWrapper.name should never be None')
        with self.pool_mutex:
            try:
                t = self.pool[socket.name]
                t.sockets.remove(socket)
                if not t.sockets:
                    t.bus.shutdown()
                    del self.pool[socket.name]
            except KeyError:
                warning('Socket %s already removed from pool' % socket.name)
SocketsPool = _SocketsPool()

class SocketWrapper(can_BusABC):
    """Helper class to wrap a python-can Bus object as socket"""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        'Initializes a new python-can based socket, described by the provided\n        arguments and keyword arguments. This SocketWrapper gets automatically\n        registered in the SocketsPool.\n\n        :param args: Arguments for the python-can Bus object\n        :param kwargs: Keyword arguments for the python-can Bus object\n        '
        super(SocketWrapper, self).__init__(*args, **kwargs)
        self.lock = threading.Lock()
        self.rx_queue = deque()
        self.name = None
        SocketsPool.register(self, *args, **kwargs)

    def _recv_internal(self, timeout):
        if False:
            print('Hello World!')
        'Internal blocking receive method,\n        following the ``can_BusABC`` interface of python-can.\n\n        This triggers the multiplex function of the general SocketsPool.\n\n        :param timeout: Time to wait for a packet\n        :return: Returns a tuple of either a can_Message or None and a bool to\n                 indicate if filtering was already applied.\n        '
        if not self.rx_queue:
            return (None, True)
        with self.lock:
            if len(self.rx_queue) == 0:
                return (None, True)
            msg = self.rx_queue.popleft()
            return (msg, True)

    def send(self, msg, timeout=None):
        if False:
            print('Hello World!')
        'Send function, following the ``can_BusABC`` interface of python-can.\n\n        :param msg: Message to be sent.\n        :param timeout: Not used.\n        '
        SocketsPool.internal_send(self, msg)

    def shutdown(self):
        if False:
            return 10
        'Shutdown function, following the ``can_BusABC`` interface of\n        python-can.\n        '
        SocketsPool.unregister(self)
        super().shutdown()

class PythonCANSocket(SuperSocket):
    """Initializes a python-can bus object as Scapy PythonCANSocket.

    All provided keyword arguments, except *basecls* are forwarded to
    the python-can can_Bus init function. For further details on python-can
    check: https://python-can.readthedocs.io/

    Example:
        >>> socket = PythonCANSocket(bustype='socketcan', channel='vcan0', bitrate=250000)
    """
    desc = 'read/write packets at a given CAN interface using a python-can bus object'
    nonblocking_socket = True

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        self.basecls = cast(Optional[Type[Packet]], kwargs.pop('basecls', CAN))
        self.can_iface = SocketWrapper(**kwargs)

    def recv_raw(self, x=65535):
        if False:
            i = 10
            return i + 15
        'Returns a tuple containing (cls, pkt_data, time)'
        msg = self.can_iface.recv()
        hdr = msg.is_extended_id << 31 | msg.is_remote_frame << 30 | msg.is_error_frame << 29 | msg.arbitration_id
        if conf.contribs['CAN']['swap-bytes']:
            hdr = struct.unpack('<I', struct.pack('>I', hdr))[0]
        dlc = msg.dlc << 24 | msg.is_fd << 18 | msg.error_state_indicator << 17 | msg.bitrate_switch << 16
        pkt_data = struct.pack('!II', hdr, dlc) + bytes(msg.data)
        return (self.basecls, pkt_data, msg.timestamp)

    def send(self, x):
        if False:
            i = 10
            return i + 15
        bx = bytes(x)
        msg = can_Message(is_remote_frame=x.flags == 2, is_extended_id=x.flags == 4, is_error_frame=x.flags == 1, arbitration_id=x.identifier, is_fd=bx[5] & 4 > 0, error_state_indicator=bx[5] & 2 > 0, bitrate_switch=bx[5] & 1 > 0, dlc=x.length, data=bx[8:])
        msg.timestamp = time.time()
        try:
            x.sent_time = msg.timestamp
        except AttributeError:
            pass
        self.can_iface.send(msg)
        return len(x)

    @staticmethod
    def select(sockets, remain=conf.recv_poll_rate):
        if False:
            return 10
        'This function is called during sendrecv() routine to select\n        the available sockets.\n\n        :param sockets: an array of sockets that need to be selected\n        :returns: an array of sockets that were selected and\n            the function to be called next to get the packets (i.g. recv)\n        '
        ready_sockets = [s for s in sockets if isinstance(s, PythonCANSocket) and len(s.can_iface.rx_queue)]
        if not ready_sockets:
            time.sleep(0)
        SocketsPool.multiplex_rx_packets()
        return cast(List[SuperSocket], ready_sockets)

    def close(self):
        if False:
            i = 10
            return i + 15
        'Closes this socket'
        if self.closed:
            return
        super(PythonCANSocket, self).close()
        self.can_iface.shutdown()
CANSocket = PythonCANSocket