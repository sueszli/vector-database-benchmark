"""
Packet sending and receiving libpcap/WinPcap.
"""
import os
import platform
import socket
import struct
import time
from scapy.automaton import select_objects
from scapy.compat import raw, plain_str
from scapy.config import conf
from scapy.consts import WINDOWS
from scapy.data import MTU, ETH_P_ALL
from scapy.error import Scapy_Exception, log_loading, log_runtime, warning
from scapy.interfaces import InterfaceProvider, NetworkInterface, _GlobInterfaceType, network_name
from scapy.packet import Packet
from scapy.pton_ntop import inet_ntop
from scapy.supersocket import SuperSocket
from scapy.utils import str2mac, decode_locale_str
import scapy.consts
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Type, cast
if not scapy.consts.WINDOWS:
    from fcntl import ioctl
if not hasattr(socket, 'AF_LINK'):
    socket.AF_LINK = 18
BIOCIMMEDIATE = -2147204496
PCAP_IF_UP = 2
_pcap_if_flags = ['LOOPBACK', 'UP', 'RUNNING', 'WIRELESS', 'OK', 'DISCONNECTED', 'NA']

class _L2libpcapSocket(SuperSocket):
    __slots__ = ['pcap_fd']

    def __init__(self, fd):
        if False:
            i = 10
            return i + 15
        self.pcap_fd = fd
        ll = self.pcap_fd.datalink()
        if ll in conf.l2types:
            self.cls = conf.l2types[ll]
        else:
            self.cls = conf.default_l2
            warning('Unable to guess datalink type (interface=%s linktype=%i). Using %s', self.iface, ll, self.cls.name)

    def recv_raw(self, x=MTU):
        if False:
            return 10
        '\n        Receives a packet, then returns a tuple containing\n        (cls, pkt_data, time)\n        '
        (ts, pkt) = self.pcap_fd.next()
        if pkt is None:
            return (None, None, None)
        return (self.cls, pkt, ts)

    def nonblock_recv(self, x=MTU):
        if False:
            while True:
                i = 10
        'Receives and dissect a packet in non-blocking mode.'
        self.pcap_fd.setnonblock(True)
        p = self.recv(x)
        self.pcap_fd.setnonblock(False)
        return p

    def fileno(self):
        if False:
            i = 10
            return i + 15
        return self.pcap_fd.fileno()

    @staticmethod
    def select(sockets, remain=None):
        if False:
            for i in range(10):
                print('nop')
        return select_objects(sockets, remain)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if self.closed:
            return
        self.closed = True
        if hasattr(self, 'pcap_fd'):
            self.pcap_fd.close()
if conf.use_pcap:
    if WINDOWS:
        NPCAP_PATH = os.environ['WINDIR'] + '\\System32\\Npcap'
        from scapy.libs.winpcapy import pcap_setmintocopy, pcap_getevent
    else:
        from scapy.libs.winpcapy import pcap_get_selectable_fd
    from ctypes import POINTER, byref, create_string_buffer, c_ubyte, cast as ccast
    try:
        from scapy.libs.winpcapy import PCAP_ERRBUF_SIZE, PCAP_ERROR, PCAP_ERROR_NO_SUCH_DEVICE, PCAP_ERROR_PERM_DENIED, bpf_program, pcap_close, pcap_compile, pcap_datalink, pcap_findalldevs, pcap_freealldevs, pcap_if_t, pcap_lib_version, pcap_next_ex, pcap_open_live, pcap_pkthdr, pcap_setfilter, pcap_setnonblock, sockaddr_in, sockaddr_in6
        try:
            from scapy.libs.winpcapy import pcap_inject
        except ImportError:
            from scapy.libs.winpcapy import pcap_sendpacket as pcap_inject

        def load_winpcapy():
            if False:
                print('Hello World!')
            'This functions calls libpcap ``pcap_findalldevs`` function,\n            and extracts and parse all the data scapy will need\n            to build the Interface List.\n\n            The data will be stored in ``conf.cache_pcapiflist``\n            '
            from scapy.fields import FlagValue
            err = create_string_buffer(PCAP_ERRBUF_SIZE)
            devs = POINTER(pcap_if_t)()
            if_list = {}
            if pcap_findalldevs(byref(devs), err) < 0:
                return
            try:
                p = devs
                while p:
                    name = plain_str(p.contents.name)
                    description = plain_str(p.contents.description or '')
                    flags = p.contents.flags
                    ips = []
                    mac = ''
                    a = p.contents.addresses
                    while a:
                        family = a.contents.addr.contents.sa_family
                        ap = a.contents.addr
                        if family == socket.AF_INET:
                            val = ccast(ap, POINTER(sockaddr_in))
                            addr_raw = val.contents.sin_addr[:]
                        elif family == socket.AF_INET6:
                            val = ccast(ap, POINTER(sockaddr_in6))
                            addr_raw = val.contents.sin6_addr[:]
                        elif family == socket.AF_LINK:
                            val = ap.contents.sa_data
                            mac = str2mac(bytes(bytearray(val[:6])))
                            a = a.contents.next
                            continue
                        else:
                            a = a.contents.next
                            continue
                        addr = inet_ntop(family, bytes(bytearray(addr_raw)))
                        if addr != '0.0.0.0':
                            ips.append(addr)
                        a = a.contents.next
                    flags = FlagValue(flags, _pcap_if_flags)
                    if_list[name] = (description, ips, flags, mac)
                    p = p.contents.next
                conf.cache_pcapiflist = if_list
            except Exception:
                raise
            finally:
                pcap_freealldevs(devs)
    except OSError:
        conf.use_pcap = False
        if WINDOWS:
            if conf.interactive:
                log_loading.critical('Npcap/Winpcap is not installed ! See https://scapy.readthedocs.io/en/latest/installation.html#windows')
        elif conf.interactive:
            log_loading.critical('Libpcap is not installed!')
    else:
        if WINDOWS:
            version = pcap_lib_version()
            if b'winpcap' in version.lower():
                if os.path.exists(NPCAP_PATH + '\\wpcap.dll'):
                    warning("Winpcap is installed over Npcap. Will use Winpcap (see 'Winpcap/Npcap conflicts' in Scapy's docs)")
                elif platform.release() != 'XP':
                    warning('WinPcap is now deprecated (not maintained). Please use Npcap instead')
            elif b'npcap' in version.lower():
                conf.use_npcap = True
                conf.loopback_name = conf.loopback_name = 'Npcap Loopback Adapter'
if WINDOWS:
    NPCAP_PATH = ''
if conf.use_pcap:

    class _PcapWrapper_libpcap:
        """Wrapper for the libpcap calls"""

        def __init__(self, device, snaplen, promisc, to_ms, monitor=None):
            if False:
                i = 10
                return i + 15
            self.errbuf = create_string_buffer(PCAP_ERRBUF_SIZE)
            self.iface = create_string_buffer(network_name(device).encode('utf8'))
            self.dtl = -1
            if monitor:
                if WINDOWS and (not conf.use_npcap):
                    raise OSError('On Windows, this feature requires NPcap !')
                from scapy.libs.winpcapy import pcap_create, pcap_set_snaplen, pcap_set_promisc, pcap_set_timeout, pcap_set_rfmon, pcap_activate, pcap_statustostr, pcap_geterr
                self.pcap = pcap_create(self.iface, self.errbuf)
                if not self.pcap:
                    error = decode_locale_str(bytearray(self.errbuf).strip(b'\x00'))
                    if error:
                        raise OSError(error)
                pcap_set_snaplen(self.pcap, snaplen)
                pcap_set_promisc(self.pcap, promisc)
                pcap_set_timeout(self.pcap, to_ms)
                if pcap_set_rfmon(self.pcap, 1) != 0:
                    log_runtime.error('Could not set monitor mode')
                status = pcap_activate(self.pcap)
                if status < 0:
                    iface = decode_locale_str(bytearray(self.iface).strip(b'\x00'))
                    errstr = decode_locale_str(bytearray(pcap_geterr(self.pcap)).strip(b'\x00'))
                    statusstr = decode_locale_str(bytearray(pcap_statustostr(status)).strip(b'\x00'))
                    if status == PCAP_ERROR:
                        errmsg = errstr
                    elif status == PCAP_ERROR_NO_SUCH_DEVICE:
                        errmsg = '%s: %s\n(%s)' % (iface, statusstr, errstr)
                    elif status == PCAP_ERROR_PERM_DENIED and errstr != '':
                        errmsg = '%s: %s\n(%s)' % (iface, statusstr, errstr)
                    else:
                        errmsg = '%s: %s' % (iface, statusstr)
                    raise OSError(errmsg)
            else:
                self.pcap = pcap_open_live(self.iface, snaplen, promisc, to_ms, self.errbuf)
                error = decode_locale_str(bytearray(self.errbuf).strip(b'\x00'))
                if error:
                    raise OSError(error)
            if WINDOWS:
                pcap_setmintocopy(self.pcap, 0)
            self.header = POINTER(pcap_pkthdr)()
            self.pkt_data = POINTER(c_ubyte)()
            self.bpf_program = bpf_program()

        def next(self):
            if False:
                return 10
            '\n            Returns the next packet as the tuple\n            (timestamp, raw_packet)\n            '
            c = pcap_next_ex(self.pcap, byref(self.header), byref(self.pkt_data))
            if not c > 0:
                return (None, None)
            ts = self.header.contents.ts.tv_sec + float(self.header.contents.ts.tv_usec) / 1000000.0
            pkt = bytes(bytearray(self.pkt_data[:self.header.contents.len]))
            return (ts, pkt)
        __next__ = next

        def datalink(self):
            if False:
                for i in range(10):
                    print('nop')
            'Wrapper around pcap_datalink'
            if self.dtl == -1:
                self.dtl = pcap_datalink(self.pcap)
            return self.dtl

        def fileno(self):
            if False:
                print('Hello World!')
            if WINDOWS:
                return cast(int, pcap_getevent(self.pcap))
            else:
                return cast(int, pcap_get_selectable_fd(self.pcap))

        def setfilter(self, f):
            if False:
                return 10
            filter_exp = create_string_buffer(f.encode('utf8'))
            if pcap_compile(self.pcap, byref(self.bpf_program), filter_exp, 1, -1) == -1:
                log_runtime.error('Could not compile filter expression %s', f)
                return False
            elif pcap_setfilter(self.pcap, byref(self.bpf_program)) == -1:
                log_runtime.error('Could not set filter %s', f)
                return False
            return True

        def setnonblock(self, i):
            if False:
                print('Hello World!')
            pcap_setnonblock(self.pcap, i, self.errbuf)

        def send(self, x):
            if False:
                return 10
            return pcap_inject(self.pcap, x, len(x))

        def close(self):
            if False:
                print('Hello World!')
            pcap_close(self.pcap)
    open_pcap = _PcapWrapper_libpcap

    class LibpcapProvider(InterfaceProvider):
        """
        Load interfaces from Libpcap on non-Windows machines
        """
        name = 'libpcap'
        libpcap = True

        def load(self):
            if False:
                return 10
            if not conf.use_pcap or WINDOWS:
                return {}
            if not conf.cache_pcapiflist:
                load_winpcapy()
            data = {}
            i = 0
            for (ifname, dat) in conf.cache_pcapiflist.items():
                (description, ips, flags, mac) = dat
                i += 1
                if not mac:
                    from scapy.arch import get_if_hwaddr
                    try:
                        mac = get_if_hwaddr(ifname)
                    except Exception:
                        continue
                if_data = {'name': ifname, 'description': description or ifname, 'network_name': ifname, 'index': i, 'mac': mac or '00:00:00:00:00:00', 'ips': ips, 'flags': flags}
                data[ifname] = NetworkInterface(self, if_data)
            return data

        def reload(self):
            if False:
                while True:
                    i = 10
            if conf.use_pcap:
                from scapy.arch.libpcap import load_winpcapy
                load_winpcapy()
            return self.load()
    if not WINDOWS:
        conf.ifaces.register_provider(LibpcapProvider)

    class L2pcapListenSocket(_L2libpcapSocket):
        desc = 'read packets at layer 2 using libpcap'

        def __init__(self, iface=None, type=ETH_P_ALL, promisc=None, filter=None, monitor=None):
            if False:
                return 10
            self.type = type
            self.outs = None
            if iface is None:
                iface = conf.iface
            self.iface = iface
            if promisc is not None:
                self.promisc = promisc
            else:
                self.promisc = conf.sniff_promisc
            fd = open_pcap(iface, MTU, self.promisc, 100, monitor=monitor)
            super(L2pcapListenSocket, self).__init__(fd)
            try:
                if not WINDOWS:
                    ioctl(self.pcap_fd.fileno(), BIOCIMMEDIATE, struct.pack('I', 1))
            except Exception:
                pass
            if type == ETH_P_ALL:
                if conf.except_filter:
                    if filter:
                        filter = '(%s) and not (%s)' % (filter, conf.except_filter)
                    else:
                        filter = 'not (%s)' % conf.except_filter
                if filter:
                    self.pcap_fd.setfilter(filter)

        def send(self, x):
            if False:
                return 10
            raise Scapy_Exception("Can't send anything with L2pcapListenSocket")

    class L2pcapSocket(_L2libpcapSocket):
        desc = 'read/write packets at layer 2 using only libpcap'

        def __init__(self, iface=None, type=ETH_P_ALL, promisc=None, filter=None, nofilter=0, monitor=None):
            if False:
                i = 10
                return i + 15
            if iface is None:
                iface = conf.iface
            self.iface = iface
            if promisc is not None:
                self.promisc = promisc
            else:
                self.promisc = conf.sniff_promisc
            fd = open_pcap(iface, MTU, self.promisc, 100, monitor=monitor)
            super(L2pcapSocket, self).__init__(fd)
            try:
                if not WINDOWS:
                    ioctl(self.pcap_fd.fileno(), BIOCIMMEDIATE, struct.pack('I', 1))
            except Exception:
                pass
            if nofilter:
                if type != ETH_P_ALL:
                    filter = 'ether proto %i' % type
                else:
                    filter = None
            else:
                if conf.except_filter:
                    if filter:
                        filter = '(%s) and not (%s)' % (filter, conf.except_filter)
                    else:
                        filter = 'not (%s)' % conf.except_filter
                if type != ETH_P_ALL:
                    if filter:
                        filter = '(ether proto %i) and (%s)' % (type, filter)
                    else:
                        filter = 'ether proto %i' % type
            if filter:
                self.pcap_fd.setfilter(filter)

        def send(self, x):
            if False:
                for i in range(10):
                    print('nop')
            sx = raw(x)
            try:
                x.sent_time = time.time()
            except AttributeError:
                pass
            return self.pcap_fd.send(sx)

    class L3pcapSocket(L2pcapSocket):
        desc = 'read/write packets at layer 3 using only libpcap'

        def recv(self, x=MTU, **kwargs):
            if False:
                i = 10
                return i + 15
            r = L2pcapSocket.recv(self, x, **kwargs)
            if r:
                r.payload.time = r.time
                return r.payload
            return r

        def send(self, x):
            if False:
                print('Hello World!')
            sx = raw(self.cls() / x)
            try:
                x.sent_time = time.time()
            except AttributeError:
                pass
            return self.pcap_fd.send(sx)