"""
Scapy *BSD native support - BPF sockets
"""
from select import select
import ctypes
import errno
import fcntl
import os
import platform
import struct
import sys
import time
from scapy.arch.bpf.core import get_dev_bpf, attach_filter
from scapy.arch.bpf.consts import BIOCGBLEN, BIOCGDLT, BIOCGSTATS, BIOCIMMEDIATE, BIOCPROMISC, BIOCSBLEN, BIOCSDLT, BIOCSETIF, BIOCSHDRCMPLT, BIOCSTSTAMP, BPF_BUFFER_LENGTH, BPF_T_NANOTIME
from scapy.config import conf
from scapy.consts import DARWIN, FREEBSD, NETBSD
from scapy.data import ETH_P_ALL, DLT_IEEE802_11_RADIO
from scapy.error import Scapy_Exception, warning
from scapy.interfaces import network_name
from scapy.supersocket import SuperSocket
from scapy.compat import raw
if FREEBSD or NETBSD:
    BPF_ALIGNMENT = ctypes.sizeof(ctypes.c_long)
else:
    BPF_ALIGNMENT = ctypes.sizeof(ctypes.c_int32)
_NANOTIME = FREEBSD
if _NANOTIME:

    class bpf_timeval(ctypes.Structure):
        _fields_ = [('tv_sec', ctypes.c_ulong), ('tv_nsec', ctypes.c_ulong)]
elif NETBSD:

    class bpf_timeval(ctypes.Structure):
        _fields_ = [('tv_sec', ctypes.c_ulong), ('tv_usec', ctypes.c_ulong)]
else:

    class bpf_timeval(ctypes.Structure):
        _fields_ = [('tv_sec', ctypes.c_uint32), ('tv_usec', ctypes.c_uint32)]

class bpf_hdr(ctypes.Structure):
    _fields_ = [('bh_tstamp', bpf_timeval), ('bh_caplen', ctypes.c_uint32), ('bh_datalen', ctypes.c_uint32), ('bh_hdrlen', ctypes.c_uint16)]
_bpf_hdr_len = ctypes.sizeof(bpf_hdr)

class _L2bpfSocket(SuperSocket):
    """"Generic Scapy BPF Super Socket"""
    desc = 'read/write packets using BPF'
    nonblocking_socket = True

    def __init__(self, iface=None, type=ETH_P_ALL, promisc=None, filter=None, nofilter=0, monitor=False):
        if False:
            i = 10
            return i + 15
        if monitor:
            raise Scapy_Exception('We do not natively support monitor mode on BPF. Please turn on libpcap using conf.use_pcap = True')
        self.fd_flags = None
        self.assigned_interface = None
        if promisc is None:
            promisc = conf.sniff_promisc
        self.promisc = promisc
        self.iface = network_name(iface or conf.iface)
        self.ins = None
        (self.ins, self.dev_bpf) = get_dev_bpf()
        self.outs = self.ins
        if FREEBSD:
            try:
                fcntl.ioctl(self.ins, BIOCSTSTAMP, struct.pack('I', BPF_T_NANOTIME))
            except IOError:
                raise Scapy_Exception('BIOCSTSTAMP failed on /dev/bpf%i' % self.dev_bpf)
        try:
            fcntl.ioctl(self.ins, BIOCSBLEN, struct.pack('I', BPF_BUFFER_LENGTH))
        except IOError:
            raise Scapy_Exception('BIOCSBLEN failed on /dev/bpf%i' % self.dev_bpf)
        try:
            fcntl.ioctl(self.ins, BIOCSETIF, struct.pack('16s16x', self.iface.encode()))
        except IOError:
            raise Scapy_Exception('BIOCSETIF failed on %s' % self.iface)
        self.assigned_interface = self.iface
        if self.promisc:
            self.set_promisc(1)
        if DARWIN and monitor:
            try:
                tmp_mac_version = platform.mac_ver()[0].split('.')
                tmp_mac_version = [int(num) for num in tmp_mac_version]
                macos_version = tmp_mac_version[0] * 10000
                macos_version += tmp_mac_version[1] * 100 + tmp_mac_version[2]
            except (IndexError, ValueError):
                warning('Could not determine your macOS version!')
                macos_version = sys.maxint
            if macos_version < 101500:
                dlt_radiotap = struct.pack('I', DLT_IEEE802_11_RADIO)
                try:
                    fcntl.ioctl(self.ins, BIOCSDLT, dlt_radiotap)
                except IOError:
                    raise Scapy_Exception("Can't set %s into monitor mode!" % self.iface)
            else:
                warning("Scapy won't activate 802.11 monitoring, as it will crash your macOS kernel!")
        try:
            fcntl.ioctl(self.ins, BIOCIMMEDIATE, struct.pack('I', 1))
        except IOError:
            raise Scapy_Exception('BIOCIMMEDIATE failed on /dev/bpf%i' % self.dev_bpf)
        try:
            fcntl.ioctl(self.ins, BIOCSHDRCMPLT, struct.pack('i', 1))
        except IOError:
            raise Scapy_Exception('BIOCSHDRCMPLT failed on /dev/bpf%i' % self.dev_bpf)
        filter_attached = False
        if not nofilter:
            if conf.except_filter:
                if filter:
                    filter = '(%s) and not (%s)' % (filter, conf.except_filter)
                else:
                    filter = 'not (%s)' % conf.except_filter
            if filter is not None:
                try:
                    attach_filter(self.ins, filter, self.iface)
                    filter_attached = True
                except ImportError as ex:
                    warning('Cannot set filter: %s' % ex)
        if NETBSD and filter_attached is False:
            filter = 'greater 0'
            try:
                attach_filter(self.ins, filter, self.iface)
            except ImportError as ex:
                warning('Cannot set filter: %s' % ex)
        self.guessed_cls = self.guess_cls()

    def set_promisc(self, value):
        if False:
            while True:
                i = 10
        'Set the interface in promiscuous mode'
        try:
            fcntl.ioctl(self.ins, BIOCPROMISC, struct.pack('i', value))
        except IOError:
            raise Scapy_Exception('Cannot set promiscuous mode on interface (%s)!' % self.iface)

    def __del__(self):
        if False:
            print('Hello World!')
        'Close the file descriptor on delete'
        if self is not None:
            self.close()

    def guess_cls(self):
        if False:
            while True:
                i = 10
        'Guess the packet class that must be used on the interface'
        try:
            ret = fcntl.ioctl(self.ins, BIOCGDLT, struct.pack('I', 0))
            ret = struct.unpack('I', ret)[0]
        except IOError:
            cls = conf.default_l2
            warning('BIOCGDLT failed: unable to guess type. Using %s !', cls.name)
            return cls
        try:
            return conf.l2types[ret]
        except KeyError:
            cls = conf.default_l2
            warning('Unable to guess type (type %i). Using %s', ret, cls.name)

    def set_nonblock(self, set_flag=True):
        if False:
            for i in range(10):
                print('nop')
        'Set the non blocking flag on the socket'
        if self.fd_flags is None:
            try:
                self.fd_flags = fcntl.fcntl(self.ins, fcntl.F_GETFL)
            except IOError:
                warning('Cannot get flags on this file descriptor !')
                return
        if set_flag:
            new_fd_flags = self.fd_flags | os.O_NONBLOCK
        else:
            new_fd_flags = self.fd_flags & ~os.O_NONBLOCK
        try:
            fcntl.fcntl(self.ins, fcntl.F_SETFL, new_fd_flags)
            self.fd_flags = new_fd_flags
        except Exception:
            warning("Can't set flags on this file descriptor !")

    def get_stats(self):
        if False:
            i = 10
            return i + 15
        'Get received / dropped statistics'
        try:
            ret = fcntl.ioctl(self.ins, BIOCGSTATS, struct.pack('2I', 0, 0))
            return struct.unpack('2I', ret)
        except IOError:
            warning('Unable to get stats from BPF !')
            return (None, None)

    def get_blen(self):
        if False:
            return 10
        'Get the BPF buffer length'
        try:
            ret = fcntl.ioctl(self.ins, BIOCGBLEN, struct.pack('I', 0))
            return struct.unpack('I', ret)[0]
        except IOError:
            warning('Unable to get the BPF buffer length')
            return

    def fileno(self):
        if False:
            while True:
                i = 10
        'Get the underlying file descriptor'
        return self.ins

    def close(self):
        if False:
            return 10
        'Close the Super Socket'
        if not self.closed and self.ins is not None:
            os.close(self.ins)
            self.closed = True
            self.ins = None

    def send(self, x):
        if False:
            return 10
        'Dummy send method'
        raise Exception("Can't send anything with %s" % self.__class__.__name__)

    def recv_raw(self, x=BPF_BUFFER_LENGTH):
        if False:
            i = 10
            return i + 15
        'Dummy recv method'
        raise Exception("Can't recv anything with %s" % self.__class__.__name__)

    @staticmethod
    def select(sockets, remain=None):
        if False:
            return 10
        'This function is called during sendrecv() routine to select\n        the available sockets.\n        '
        return bpf_select(sockets, remain)

class L2bpfListenSocket(_L2bpfSocket):
    """"Scapy L2 BPF Listen Super Socket"""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.received_frames = []
        super(L2bpfListenSocket, self).__init__(*args, **kwargs)

    def buffered_frames(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the number of frames in the buffer'
        return len(self.received_frames)

    def get_frame(self):
        if False:
            print('Hello World!')
        'Get a frame or packet from the received list'
        if self.received_frames:
            return self.received_frames.pop(0)
        else:
            return (None, None, None)

    @staticmethod
    def bpf_align(bh_h, bh_c):
        if False:
            print('Hello World!')
        'Return the index to the end of the current packet'
        return bh_h + bh_c + (BPF_ALIGNMENT - 1) & ~(BPF_ALIGNMENT - 1)

    def extract_frames(self, bpf_buffer):
        if False:
            while True:
                i = 10
        '\n        Extract all frames from the buffer and stored them in the received list\n        '
        len_bb = len(bpf_buffer)
        if len_bb < _bpf_hdr_len:
            return
        bh_hdr = bpf_hdr.from_buffer_copy(bpf_buffer)
        if bh_hdr.bh_datalen == 0:
            return
        frame_str = bpf_buffer[bh_hdr.bh_hdrlen:bh_hdr.bh_hdrlen + bh_hdr.bh_caplen]
        if _NANOTIME:
            ts = bh_hdr.bh_tstamp.tv_sec + 1e-09 * bh_hdr.bh_tstamp.tv_nsec
        else:
            ts = bh_hdr.bh_tstamp.tv_sec + 1e-06 * bh_hdr.bh_tstamp.tv_usec
        self.received_frames.append((self.guessed_cls, frame_str, ts))
        end = self.bpf_align(bh_hdr.bh_hdrlen, bh_hdr.bh_caplen)
        if len_bb - end >= 20:
            self.extract_frames(bpf_buffer[end:])

    def recv_raw(self, x=BPF_BUFFER_LENGTH):
        if False:
            return 10
        'Receive a frame from the network'
        x = min(x, BPF_BUFFER_LENGTH)
        if self.buffered_frames():
            return self.get_frame()
        try:
            bpf_buffer = os.read(self.ins, x)
        except EnvironmentError as exc:
            if exc.errno != errno.EAGAIN:
                warning('BPF recv_raw()', exc_info=True)
            return (None, None, None)
        self.extract_frames(bpf_buffer)
        return self.get_frame()

class L2bpfSocket(L2bpfListenSocket):
    """"Scapy L2 BPF Super Socket"""

    def send(self, x):
        if False:
            i = 10
            return i + 15
        'Send a frame'
        return os.write(self.outs, raw(x))

    def nonblock_recv(self):
        if False:
            while True:
                i = 10
        'Non blocking receive'
        if self.buffered_frames():
            return L2bpfListenSocket.recv(self)
        self.set_nonblock(True)
        pkt = L2bpfListenSocket.recv(self)
        self.set_nonblock(False)
        return pkt

class L3bpfSocket(L2bpfSocket):

    def recv(self, x=BPF_BUFFER_LENGTH, **kwargs):
        if False:
            return 10
        'Receive on layer 3'
        r = SuperSocket.recv(self, x, **kwargs)
        if r:
            r.payload.time = r.time
            return r.payload
        return r

    def send(self, pkt):
        if False:
            i = 10
            return i + 15
        'Send a packet'
        from scapy.layers.l2 import Loopback
        iff = pkt.route()[0]
        if iff is None:
            iff = network_name(conf.iface)
        if self.assigned_interface != iff:
            try:
                fcntl.ioctl(self.outs, BIOCSETIF, struct.pack('16s16x', iff.encode()))
            except IOError:
                raise Scapy_Exception('BIOCSETIF failed on %s' % iff)
            self.assigned_interface = iff
        if DARWIN and iff.startswith('tun') and (self.guessed_cls == Loopback):
            frame = raw(pkt)
        elif FREEBSD and (iff.startswith('tun') or iff.startswith('tap')):
            warning('Cannot write to %s according to the documentation!', iff)
            return
        else:
            frame = raw(self.guessed_cls() / pkt)
        pkt.sent_time = time.time()
        L2bpfSocket.send(self, frame)

def isBPFSocket(obj):
    if False:
        while True:
            i = 10
    'Return True is obj is a BPF Super Socket'
    return isinstance(obj, (L2bpfListenSocket, L2bpfListenSocket, L3bpfSocket))

def bpf_select(fds_list, timeout=None):
    if False:
        print('Hello World!')
    'A call to recv() can return several frames. This functions hides the fact\n       that some frames are read from the internal buffer.'
    bpf_scks_buffered = list()
    select_fds = list()
    for tmp_fd in fds_list:
        if isBPFSocket(tmp_fd) and tmp_fd.buffered_frames():
            bpf_scks_buffered.append(tmp_fd)
            continue
        select_fds.append(tmp_fd)
    if select_fds:
        if timeout is None:
            timeout = 0.05
        (ready_list, _, _) = select(select_fds, [], [], timeout)
        return bpf_scks_buffered + ready_list
    else:
        return bpf_scks_buffered