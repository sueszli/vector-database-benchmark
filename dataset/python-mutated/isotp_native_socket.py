import ctypes
from ctypes.util import find_library
import struct
import socket
from scapy.contrib.isotp import log_isotp
from scapy.packet import Packet
from scapy.error import Scapy_Exception
from scapy.supersocket import SuperSocket
from scapy.data import SO_TIMESTAMPNS
from scapy.config import conf
from scapy.arch.linux import get_last_packet_timestamp, SIOCGIFINDEX
from scapy.contrib.isotp.isotp_packet import ISOTP
from scapy.layers.can import CAN_MTU, CAN_FD_MTU, CAN_MAX_DLEN, CAN_FD_MAX_DLEN
from typing import Any, Optional, Union, Tuple, Type, cast
LIBC = ctypes.cdll.LoadLibrary(find_library('c'))
CAN_ISOTP = 6
SOL_CAN_BASE = 100
SOL_CAN_ISOTP = SOL_CAN_BASE + CAN_ISOTP
CAN_ISOTP_OPTS = 1
CAN_ISOTP_RECV_FC = 2
CAN_ISOTP_TX_STMIN = 3
CAN_ISOTP_RX_STMIN = 4
CAN_ISOTP_LL_OPTS = 5
CAN_ISOTP_LISTEN_MODE = 1
CAN_ISOTP_EXTEND_ADDR = 2
CAN_ISOTP_TX_PADDING = 4
CAN_ISOTP_RX_PADDING = 8
CAN_ISOTP_CHK_PAD_LEN = 16
CAN_ISOTP_CHK_PAD_DATA = 32
CAN_ISOTP_HALF_DUPLEX = 64
CAN_ISOTP_FORCE_TXSTMIN = 128
CAN_ISOTP_FORCE_RXSTMIN = 256
CAN_ISOTP_RX_EXT_ADDR = 512
CAN_ISOTP_DEFAULT_FLAGS = 0
CAN_ISOTP_DEFAULT_EXT_ADDRESS = 0
CAN_ISOTP_DEFAULT_PAD_CONTENT = 204
CAN_ISOTP_DEFAULT_FRAME_TXTIME = 0
CAN_ISOTP_DEFAULT_RECV_BS = 0
CAN_ISOTP_DEFAULT_RECV_STMIN = 0
CAN_ISOTP_DEFAULT_RECV_WFTMAX = 0
CAN_ISOTP_DEFAULT_LL_MTU = CAN_MTU
CAN_ISOTP_CANFD_MTU = CAN_FD_MTU
CAN_ISOTP_DEFAULT_LL_TX_DL = CAN_MAX_DLEN
CAN_FD_ISOTP_DEFAULT_LL_TX_DL = CAN_FD_MAX_DLEN
CAN_ISOTP_DEFAULT_LL_TX_FLAGS = 0

class tp(ctypes.Structure):
    _fields_ = [('rx_id', ctypes.c_uint32), ('tx_id', ctypes.c_uint32)]

class addr_info(ctypes.Union):
    _fields_ = [('tp', tp)]

class sockaddr_can(ctypes.Structure):
    _fields_ = [('can_family', ctypes.c_uint16), ('can_ifindex', ctypes.c_int), ('can_addr', addr_info)]

class ifreq(ctypes.Structure):
    _fields_ = [('ifr_name', ctypes.c_char * 16), ('ifr_ifindex', ctypes.c_int)]

class ISOTPNativeSocket(SuperSocket):
    """
    ISOTPSocket using the can-isotp kernel module

    :param iface: a CANSocket instance or an interface name
    :param tx_id: the CAN identifier of the sent CAN frames
    :param rx_id: the CAN identifier of the received CAN frames
    :param ext_address: the extended address of the sent ISOTP frames
    :param rx_ext_address: the extended address of the received ISOTP frames
    :param bs: block size sent in Flow Control ISOTP frames
    :param stmin: minimum desired separation time sent in
                  Flow Control ISOTP frames
    :param padding: If True, pads sending packets with 0x00 which not
                    count to the payload.
                    Does not affect receiving packets.
    :param listen_only: Does not send Flow Control frames if a First Frame is
                        received
    :param frame_txtime: Separation time between two CAN frames during send
    :param basecls: base class of the packets emitted by this socket
    """
    desc = 'read/write packets at a given CAN interface using CAN_ISOTP socket '
    can_isotp_options_fmt = '@2I4B'
    can_isotp_fc_options_fmt = '@3B'
    can_isotp_ll_options_fmt = '@3B'
    sockaddr_can_fmt = '@H3I'
    auxdata_available = True

    def __build_can_isotp_options(self, flags=CAN_ISOTP_DEFAULT_FLAGS, frame_txtime=CAN_ISOTP_DEFAULT_FRAME_TXTIME, ext_address=CAN_ISOTP_DEFAULT_EXT_ADDRESS, txpad_content=CAN_ISOTP_DEFAULT_PAD_CONTENT, rxpad_content=CAN_ISOTP_DEFAULT_PAD_CONTENT, rx_ext_address=CAN_ISOTP_DEFAULT_EXT_ADDRESS):
        if False:
            print('Hello World!')
        return struct.pack(self.can_isotp_options_fmt, flags, frame_txtime, ext_address, txpad_content, rxpad_content, rx_ext_address)

    def __build_can_isotp_fc_options(self, bs=CAN_ISOTP_DEFAULT_RECV_BS, stmin=CAN_ISOTP_DEFAULT_RECV_STMIN, wftmax=CAN_ISOTP_DEFAULT_RECV_WFTMAX):
        if False:
            for i in range(10):
                print('nop')
        return struct.pack(self.can_isotp_fc_options_fmt, bs, stmin, wftmax)

    def __build_can_isotp_ll_options(self, mtu=CAN_ISOTP_DEFAULT_LL_MTU, tx_dl=CAN_ISOTP_DEFAULT_LL_TX_DL, tx_flags=CAN_ISOTP_DEFAULT_LL_TX_FLAGS):
        if False:
            return 10
        return struct.pack(self.can_isotp_ll_options_fmt, mtu, tx_dl, tx_flags)

    def __get_sock_ifreq(self, sock, iface):
        if False:
            for i in range(10):
                print('nop')
        socket_id = ctypes.c_int(sock.fileno())
        ifr = ifreq()
        ifr.ifr_name = iface.encode('ascii')
        ret = LIBC.ioctl(socket_id, SIOCGIFINDEX, ctypes.byref(ifr))
        if ret < 0:
            m = u'Failure while getting "{}" interface index.'.format(iface)
            raise Scapy_Exception(m)
        return ifr

    def __bind_socket(self, sock, iface, tx_id, rx_id):
        if False:
            print('Hello World!')
        socket_id = ctypes.c_int(sock.fileno())
        ifr = self.__get_sock_ifreq(sock, iface)
        if tx_id > 2047:
            tx_id = tx_id | socket.CAN_EFF_FLAG
        if rx_id > 2047:
            rx_id = rx_id | socket.CAN_EFF_FLAG
        addr = sockaddr_can(ctypes.c_uint16(socket.PF_CAN), ifr.ifr_ifindex, addr_info(tp(ctypes.c_uint32(rx_id), ctypes.c_uint32(tx_id))))
        error = LIBC.bind(socket_id, ctypes.byref(addr), ctypes.sizeof(addr))
        if error < 0:
            log_isotp.warning("Couldn't bind socket")

    def __set_option_flags(self, sock, extended_addr=None, extended_rx_addr=None, listen_only=False, padding=False, transmit_time=100):
        if False:
            i = 10
            return i + 15
        option_flags = CAN_ISOTP_DEFAULT_FLAGS
        if extended_addr is not None:
            option_flags = option_flags | CAN_ISOTP_EXTEND_ADDR
        else:
            extended_addr = CAN_ISOTP_DEFAULT_EXT_ADDRESS
        if extended_rx_addr is not None:
            option_flags = option_flags | CAN_ISOTP_RX_EXT_ADDR
        else:
            extended_rx_addr = CAN_ISOTP_DEFAULT_EXT_ADDRESS
        if listen_only:
            option_flags = option_flags | CAN_ISOTP_LISTEN_MODE
        if padding:
            option_flags = option_flags | CAN_ISOTP_TX_PADDING | CAN_ISOTP_RX_PADDING
        sock.setsockopt(SOL_CAN_ISOTP, CAN_ISOTP_OPTS, self.__build_can_isotp_options(frame_txtime=transmit_time, flags=option_flags, ext_address=extended_addr, rx_ext_address=extended_rx_addr))

    def __init__(self, iface=None, tx_id=0, rx_id=0, ext_address=None, rx_ext_address=None, bs=CAN_ISOTP_DEFAULT_RECV_BS, stmin=CAN_ISOTP_DEFAULT_RECV_STMIN, padding=False, listen_only=False, frame_txtime=CAN_ISOTP_DEFAULT_FRAME_TXTIME, fd=False, basecls=ISOTP):
        if False:
            return 10
        if not isinstance(iface, str):
            iface = cast(SuperSocket, iface)
            if hasattr(iface, 'ins') and hasattr(iface.ins, 'getsockname'):
                iface = iface.ins.getsockname()
                if isinstance(iface, tuple):
                    iface = cast(str, iface[0])
            else:
                raise Scapy_Exception('Provide a string or a CANSocket object as iface parameter')
        self.iface = cast(str, iface) or conf.contribs['NativeCANSocket']['iface']
        self.can_socket = socket.socket(socket.PF_CAN, socket.SOCK_DGRAM, CAN_ISOTP)
        self.__set_option_flags(self.can_socket, ext_address, rx_ext_address, listen_only, padding, frame_txtime)
        self.tx_id = tx_id
        self.rx_id = rx_id
        self.ext_address = ext_address
        self.rx_ext_address = rx_ext_address
        self.can_socket.setsockopt(SOL_CAN_ISOTP, CAN_ISOTP_RECV_FC, self.__build_can_isotp_fc_options(stmin=stmin, bs=bs))
        self.can_socket.setsockopt(SOL_CAN_ISOTP, CAN_ISOTP_LL_OPTS, self.__build_can_isotp_ll_options(mtu=CAN_ISOTP_CANFD_MTU if fd else CAN_ISOTP_DEFAULT_LL_MTU, tx_dl=CAN_FD_ISOTP_DEFAULT_LL_TX_DL if fd else CAN_ISOTP_DEFAULT_LL_TX_DL))
        self.can_socket.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
        self.__bind_socket(self.can_socket, self.iface, tx_id, rx_id)
        self.ins = self.can_socket
        self.outs = self.can_socket
        if basecls is None:
            log_isotp.warning('Provide a basecls ')
        self.basecls = basecls

    def recv_raw(self, x=65535):
        if False:
            i = 10
            return i + 15
        '\n        Receives a packet, then returns a tuple containing\n        (cls, pkt_data, time)\n        '
        try:
            (pkt, _, ts) = self._recv_raw(self.ins, x)
        except BlockingIOError:
            log_isotp.warning('Captured no data, socket in non-blocking mode.')
            return (None, None, None)
        except socket.timeout:
            log_isotp.warning('Captured no data, socket read timed out.')
            return (None, None, None)
        except OSError as e:
            log_isotp.warning('Captured no data. %s' % e)
            if e.errno == 84:
                log_isotp.warning('Maybe a consecutive frame was missed. Increasing `stmin` could solve this problem.')
            elif e.errno == 110:
                log_isotp.warning('Captured no data, socket read timed out.')
            else:
                self.close()
            return (None, None, None)
        if ts is None:
            ts = get_last_packet_timestamp(self.ins)
        return (self.basecls, pkt, ts)

    def recv(self, x=65535, **kwargs):
        if False:
            print('Hello World!')
        msg = SuperSocket.recv(self, x, **kwargs)
        if msg is None:
            return msg
        if hasattr(msg, 'tx_id'):
            msg.tx_id = self.tx_id
        if hasattr(msg, 'rx_id'):
            msg.rx_id = self.rx_id
        if hasattr(msg, 'ext_address'):
            msg.ext_address = self.ext_address
        if hasattr(msg, 'rx_ext_address'):
            msg.rx_ext_address = self.rx_ext_address
        return msg