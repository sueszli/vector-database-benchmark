"""
Functions to send and receive packets.
"""
import itertools
from threading import Thread, Event
import os
import re
import socket
import subprocess
import time
from scapy.compat import plain_str
from scapy.data import ETH_P_ALL
from scapy.config import conf
from scapy.error import warning
from scapy.interfaces import network_name, resolve_iface, NetworkInterface
from scapy.packet import Packet
from scapy.pton_ntop import inet_pton
from scapy.utils import get_temp_file, tcpdump, wrpcap, ContextManagerSubprocess, PcapReader, EDecimal
from scapy.plist import PacketList, QueryAnswer, SndRcvList
from scapy.error import log_runtime, log_interactive, Scapy_Exception
from scapy.base_classes import Gen, SetGen
from scapy.sessions import DefaultSession
from scapy.supersocket import SuperSocket, IterSocket
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union, cast
from scapy.interfaces import _GlobInterfaceType
from scapy.plist import _PacketIterable
if conf.route is None:
    import scapy.route

class debug:
    recv = PacketList([], 'Received')
    sent = PacketList([], 'Sent')
    match = SndRcvList([], 'Matched')
    crashed_on = None
_DOC_SNDRCV_PARAMS = '\n    :param pks: SuperSocket instance to send/receive packets\n    :param pkt: the packet to send\n    :param timeout: how much time to wait after the last packet has been sent\n    :param inter: delay between two packets during sending\n    :param verbose: set verbosity level\n    :param chainCC: if True, KeyboardInterrupts will be forwarded\n    :param retry: if positive, how many times to resend unanswered packets\n        if negative, how many times to retry when no more packets\n        are answered\n    :param multi: whether to accept multiple answers for the same stimulus\n    :param rcv_pks: if set, will be used instead of pks to receive packets.\n        packets will still be sent through pks\n    :param prebuild: pre-build the packets before starting to send them.\n        Automatically enabled when a generator is passed as the packet\n    :param _flood:\n    :param threaded: if True, packets will be sent in an individual thread\n    :param session: a flow decoder used to handle stream of packets\n    :param chainEX: if True, exceptions during send will be forwarded\n    '
_GlobSessionType = Union[Type[DefaultSession], DefaultSession]

class SndRcvHandler(object):
    """
    Util to send/receive packets, used by sr*().
    Do not use directly.

    This matches the requests and answers.

    Notes::
      - threaded mode: enabling threaded mode will likely
        break packet timestamps, but might result in a speedup
        when sending a big amount of packets. Disabled by default
      - DEVS: store the outgoing timestamp right BEFORE sending the packet
        to avoid races that could result in negative latency. We aren't Stadia
    """

    def __init__(self, pks, pkt, timeout=None, inter=0, verbose=None, chainCC=False, retry=0, multi=False, rcv_pks=None, prebuild=False, _flood=None, threaded=False, session=None, chainEX=False):
        if False:
            i = 10
            return i + 15
        if verbose is None:
            verbose = conf.verb
        if conf.debug_match:
            debug.recv = PacketList([], 'Received')
            debug.sent = PacketList([], 'Sent')
            debug.match = SndRcvList([], 'Matched')
        self.nbrecv = 0
        self.ans = []
        self.pks = pks
        self.rcv_pks = rcv_pks or pks
        self.inter = inter
        self.verbose = verbose
        self.chainCC = chainCC
        self.multi = multi
        self.timeout = timeout
        self.session = session
        self.chainEX = chainEX
        self._send_done = False
        self.notans = 0
        self.noans = 0
        self._flood = _flood
        if prebuild and (not self._flood):
            self.tobesent = list(pkt)
        else:
            self.tobesent = pkt
        if retry < 0:
            autostop = retry = -retry
        else:
            autostop = 0
        if timeout is not None and timeout < 0:
            self.timeout = None
        while retry >= 0:
            self.hsent = {}
            if threaded or self._flood:
                snd_thread = Thread(target=self._sndrcv_snd)
                snd_thread.daemon = True
                self._sndrcv_rcv(snd_thread.start)
                if self._flood:
                    self._flood.stop()
                snd_thread.join()
            else:
                self._sndrcv_rcv(self._sndrcv_snd)
            if multi:
                remain = [p for p in itertools.chain(*self.hsent.values()) if not hasattr(p, '_answered')]
            else:
                remain = list(itertools.chain(*self.hsent.values()))
            if autostop and len(remain) > 0 and (len(remain) != len(self.tobesent)):
                retry = autostop
            self.tobesent = remain
            if len(self.tobesent) == 0:
                break
            retry -= 1
        if conf.debug_match:
            debug.sent = PacketList(remain[:], 'Sent')
            debug.match = SndRcvList(self.ans[:])
        if multi:
            for (snd, _) in self.ans:
                if hasattr(snd, '_answered'):
                    del snd._answered
        if verbose:
            print('\nReceived %i packets, got %i answers, remaining %i packets' % (self.nbrecv + len(self.ans), len(self.ans), max(0, self.notans - self.noans)))
        self.ans_result = SndRcvList(self.ans)
        self.unans_result = PacketList(remain, 'Unanswered')

    def results(self):
        if False:
            i = 10
            return i + 15
        return (self.ans_result, self.unans_result)

    def _sndrcv_snd(self):
        if False:
            print('Hello World!')
        'Function used in the sending thread of sndrcv()'
        i = 0
        p = None
        try:
            if self.verbose:
                print('Begin emission:')
            for p in self.tobesent:
                self.hsent.setdefault(p.hashret(), []).append(p)
                self.pks.send(p)
                time.sleep(self.inter)
                i += 1
            if self.verbose:
                print('Finished sending %i packets.' % i)
        except SystemExit:
            pass
        except Exception:
            if self.chainEX:
                raise
            else:
                log_runtime.exception('--- Error sending packets')
        finally:
            try:
                cast(Packet, self.tobesent).sent_time = cast(Packet, p).sent_time
            except AttributeError:
                pass
            if self._flood:
                self.notans = self._flood.iterlen
            elif not self._send_done:
                self.notans = i
            self._send_done = True

    def _process_packet(self, r):
        if False:
            for i in range(10):
                print('nop')
        'Internal function used to process each packet.'
        if r is None:
            return
        ok = False
        h = r.hashret()
        if h in self.hsent:
            hlst = self.hsent[h]
            for (i, sentpkt) in enumerate(hlst):
                if r.answers(sentpkt):
                    self.ans.append(QueryAnswer(sentpkt, r))
                    if self.verbose > 1:
                        os.write(1, b'*')
                    ok = True
                    if not self.multi:
                        del hlst[i]
                        self.noans += 1
                    else:
                        if not hasattr(sentpkt, '_answered'):
                            self.noans += 1
                        sentpkt._answered = 1
                    break
        if self._send_done and self.noans >= self.notans and (not self.multi):
            if self.sniffer:
                self.sniffer.stop(join=False)
        if not ok:
            if self.verbose > 1:
                os.write(1, b'.')
            self.nbrecv += 1
            if conf.debug_match:
                debug.recv.append(r)

    def _sndrcv_rcv(self, callback):
        if False:
            return 10
        'Function used to receive packets and check their hashret'
        self.sniffer = None
        try:
            self.sniffer = AsyncSniffer()
            self.sniffer._run(prn=self._process_packet, timeout=self.timeout, store=False, opened_socket=self.rcv_pks, session=self.session, started_callback=callback)
        except KeyboardInterrupt:
            if self.chainCC:
                raise

def sndrcv(*args, **kwargs):
    if False:
        while True:
            i = 10
    'Scapy raw function to send a packet and receive its answer.\n    WARNING: This is an internal function. Using sr/srp/sr1/srp is\n    more appropriate in many cases.\n    '
    sndrcver = SndRcvHandler(*args, **kwargs)
    return sndrcver.results()

def __gen_send(s, x, inter=0, loop=0, count=None, verbose=None, realtime=False, return_packets=False, *args, **kargs):
    if False:
        while True:
            i = 10
    '\n    An internal function used by send/sendp to actually send the packets,\n    implement the send logic...\n\n    It will take care of iterating through the different packets\n    '
    if isinstance(x, str):
        x = conf.raw_layer(load=x)
    if not isinstance(x, Gen):
        x = SetGen(x)
    if verbose is None:
        verbose = conf.verb
    n = 0
    if count is not None:
        loop = -count
    elif not loop:
        loop = -1
    sent_packets = PacketList() if return_packets else None
    p = None
    try:
        while loop:
            dt0 = None
            for p in x:
                if realtime:
                    ct = time.time()
                    if dt0:
                        st = dt0 + float(p.time) - ct
                        if st > 0:
                            time.sleep(st)
                    else:
                        dt0 = ct - float(p.time)
                s.send(p)
                if sent_packets is not None:
                    sent_packets.append(p)
                n += 1
                if verbose:
                    os.write(1, b'.')
                time.sleep(inter)
            if loop < 0:
                loop += 1
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cast(Packet, x).sent_time = cast(Packet, p).sent_time
        except AttributeError:
            pass
    if verbose:
        print('\nSent %i packets.' % n)
    return sent_packets

def _send(x, _func, inter=0, loop=0, iface=None, count=None, verbose=None, realtime=False, return_packets=False, socket=None, **kargs):
    if False:
        for i in range(10):
            print('nop')
    'Internal function used by send and sendp'
    need_closing = socket is None
    iface = resolve_iface(iface or conf.iface)
    socket = socket or _func(iface)(iface=iface, **kargs)
    results = __gen_send(socket, x, inter=inter, loop=loop, count=count, verbose=verbose, realtime=realtime, return_packets=return_packets)
    if need_closing:
        socket.close()
    return results

@conf.commands.register
def send(x, iface=None, **kargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Send packets at layer 3\n\n    :param x: the packets\n    :param inter: time (in s) between two packets (default 0)\n    :param loop: send packet indefinitely (default 0)\n    :param count: number of packets to send (default None=1)\n    :param verbose: verbose mode (default None=conf.verb)\n    :param realtime: check that a packet was sent before sending the next one\n    :param return_packets: return the sent packets\n    :param socket: the socket to use (default is conf.L3socket(kargs))\n    :param iface: the interface to send the packets on\n    :param monitor: (not on linux) send in monitor mode\n    :returns: None\n    '
    (iface, ipv6) = _interface_selection(iface, x)
    return _send(x, lambda iface: iface.l3socket(ipv6), iface=iface, **kargs)

@conf.commands.register
def sendp(x, iface=None, iface_hint=None, socket=None, **kargs):
    if False:
        print('Hello World!')
    '\n    Send packets at layer 2\n\n    :param x: the packets\n    :param inter: time (in s) between two packets (default 0)\n    :param loop: send packet indefinitely (default 0)\n    :param count: number of packets to send (default None=1)\n    :param verbose: verbose mode (default None=conf.verb)\n    :param realtime: check that a packet was sent before sending the next one\n    :param return_packets: return the sent packets\n    :param socket: the socket to use (default is conf.L3socket(kargs))\n    :param iface: the interface to send the packets on\n    :param monitor: (not on linux) send in monitor mode\n    :returns: None\n    '
    if iface is None and iface_hint is not None and (socket is None):
        iface = conf.route.route(iface_hint)[0]
    return _send(x, lambda iface: iface.l2socket(), iface=iface, socket=socket, **kargs)

@conf.commands.register
def sendpfast(x: _PacketIterable, pps: Optional[float]=None, mbps: Optional[float]=None, realtime: bool=False, count: Optional[int]=None, loop: int=0, file_cache: bool=False, iface: Optional[_GlobInterfaceType]=None, replay_args: Optional[List[str]]=None, parse_results: bool=False):
    if False:
        for i in range(10):
            print('nop')
    "Send packets at layer 2 using tcpreplay for performance\n\n    :param pps:  packets per second\n    :param mbps: MBits per second\n    :param realtime: use packet's timestamp, bending time with real-time value\n    :param loop: send the packet indefinitely (default 0)\n    :param count: number of packets to send (default None=1)\n    :param file_cache: cache packets in RAM instead of reading from\n        disk at each iteration\n    :param iface: output interface\n    :param replay_args: List of additional tcpreplay args (List[str])\n    :param parse_results: Return a dictionary of information\n        outputted by tcpreplay (default=False)\n    :returns: stdout, stderr, command used\n    "
    if iface is None:
        iface = conf.iface
    argv = [conf.prog.tcpreplay, '--intf1=%s' % network_name(iface)]
    if pps is not None:
        argv.append('--pps=%f' % pps)
    elif mbps is not None:
        argv.append('--mbps=%f' % mbps)
    elif realtime is not None:
        argv.append('--multiplier=%f' % realtime)
    else:
        argv.append('--topspeed')
    if count:
        assert not loop, "Can't use loop and count at the same time in sendpfast"
        argv.append('--loop=%i' % count)
    elif loop:
        argv.append('--loop=0')
    if file_cache:
        argv.append('--preload-pcap')
    if replay_args is not None:
        argv.extend(replay_args)
    f = get_temp_file()
    argv.append(f)
    wrpcap(f, x)
    results = None
    with ContextManagerSubprocess(conf.prog.tcpreplay):
        try:
            cmd = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except KeyboardInterrupt:
            log_interactive.info('Interrupted by user')
        except Exception:
            os.unlink(f)
            raise
        else:
            (stdout, stderr) = cmd.communicate()
            if stderr:
                log_runtime.warning(stderr.decode())
            if parse_results:
                results = _parse_tcpreplay_result(stdout, stderr, argv)
            elif conf.verb > 2:
                log_runtime.info(stdout.decode())
    if os.path.exists(f):
        os.unlink(f)
    return results

def _parse_tcpreplay_result(stdout_b, stderr_b, argv):
    if False:
        i = 10
        return i + 15
    '\n    Parse the output of tcpreplay and modify the results_dict to populate output information.  # noqa: E501\n    Tested with tcpreplay v3.4.4\n    Tested with tcpreplay v4.1.2\n    :param stdout: stdout of tcpreplay subprocess call\n    :param stderr: stderr of tcpreplay subprocess call\n    :param argv: the command used in the subprocess call\n    :return: dictionary containing the results\n    '
    try:
        results = {}
        stdout = plain_str(stdout_b).lower()
        stderr = plain_str(stderr_b).strip().split('\n')
        elements = {'actual': (int, int, float), 'rated': (float, float, float), 'flows': (int, float, int, int), 'attempted': (int,), 'successful': (int,), 'failed': (int,), 'truncated': (int,), 'retried packets (eno': (int,), 'retried packets (eag': (int,)}
        multi = {'actual': ('packets', 'bytes', 'time'), 'rated': ('bps', 'mbps', 'pps'), 'flows': ('flows', 'fps', 'flow_packets', 'non_flow'), 'retried packets (eno': ('retried_enobufs',), 'retried packets (eag': ('retried_eagain',)}
        float_reg = '([0-9]*\\.[0-9]+|[0-9]+)'
        int_reg = '([0-9]+)'
        any_reg = '[^0-9]*'
        r_types = {int: int_reg, float: float_reg}
        for line in stdout.split('\n'):
            line = line.strip()
            for (elt, _types) in elements.items():
                if line.startswith(elt):
                    regex = any_reg.join([r_types[x] for x in _types])
                    matches = re.search(regex, line)
                    for (i, typ) in enumerate(_types):
                        name = multi.get(elt, [elt])[i]
                        if matches:
                            results[name] = typ(matches.group(i + 1))
        results['command'] = ' '.join(argv)
        results['warnings'] = stderr[:-1]
        return results
    except Exception as parse_exception:
        if not conf.interactive:
            raise
        log_runtime.error('Error parsing output: %s', parse_exception)
        return {}

def _interface_selection(iface, packet):
    if False:
        i = 10
        return i + 15
    '\n    Select the network interface according to the layer 3 destination\n    '
    (_iff, src, _) = next(packet.__iter__()).route()
    ipv6 = False
    if src:
        try:
            inet_pton(socket.AF_INET6, src)
            ipv6 = True
        except OSError:
            pass
    if iface is None:
        try:
            iff = resolve_iface(_iff or conf.iface)
        except AttributeError:
            iff = None
        return (iff or conf.iface, ipv6)
    return (resolve_iface(iface), ipv6)

@conf.commands.register
def sr(x, promisc=None, filter=None, iface=None, nofilter=0, *args, **kargs):
    if False:
        return 10
    '\n    Send and receive packets at layer 3\n    '
    (iface, ipv6) = _interface_selection(iface, x)
    s = iface.l3socket(ipv6)(promisc=promisc, filter=filter, iface=iface, nofilter=nofilter)
    result = sndrcv(s, x, *args, **kargs)
    s.close()
    return result

@conf.commands.register
def sr1(*args, **kargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Send packets at layer 3 and return only the first answer\n    '
    (ans, _) = sr(*args, **kargs)
    if ans:
        return cast(Packet, ans[0][1])
    return None

@conf.commands.register
def srp(x, promisc=None, iface=None, iface_hint=None, filter=None, nofilter=0, type=ETH_P_ALL, *args, **kargs):
    if False:
        i = 10
        return i + 15
    '\n    Send and receive packets at layer 2\n    '
    if iface is None and iface_hint is not None:
        iface = conf.route.route(iface_hint)[0]
    iface = resolve_iface(iface or conf.iface)
    s = iface.l2socket()(promisc=promisc, iface=iface, filter=filter, nofilter=nofilter, type=type)
    result = sndrcv(s, x, *args, **kargs)
    s.close()
    return result

@conf.commands.register
def srp1(*args, **kargs):
    if False:
        i = 10
        return i + 15
    '\n    Send and receive packets at layer 2 and return only the first answer\n    '
    (ans, _) = srp(*args, **kargs)
    if len(ans) > 0:
        return cast(Packet, ans[0][1])
    return None
for sr_func in [srp, srp1, sr, sr1]:
    if sr_func.__doc__ is not None:
        sr_func.__doc__ += _DOC_SNDRCV_PARAMS

def __sr_loop(srfunc, pkts, prn=lambda x: x[1].summary(), prnfail=lambda x: x.summary(), inter=1, timeout=None, count=None, verbose=None, store=1, *args, **kargs):
    if False:
        for i in range(10):
            print('nop')
    n = 0
    r = 0
    ct = conf.color_theme
    if verbose is None:
        verbose = conf.verb
    parity = 0
    ans = []
    unans = []
    if timeout is None:
        timeout = min(2 * inter, 5)
    try:
        while True:
            parity ^= 1
            col = [ct.even, ct.odd][parity]
            if count is not None:
                if count == 0:
                    break
                count -= 1
            start = time.monotonic()
            if verbose > 1:
                print('\rsend...\r', end=' ')
            res = srfunc(pkts, *args, timeout=timeout, verbose=0, chainCC=True, **kargs)
            n += len(res[0]) + len(res[1])
            r += len(res[0])
            if verbose > 1 and prn and (len(res[0]) > 0):
                msg = 'RECV %i:' % len(res[0])
                print('\r' + ct.success(msg), end=' ')
                for rcv in res[0]:
                    print(col(prn(rcv)))
                    print(' ' * len(msg), end=' ')
            if verbose > 1 and prnfail and (len(res[1]) > 0):
                msg = 'fail %i:' % len(res[1])
                print('\r' + ct.fail(msg), end=' ')
                for fail in res[1]:
                    print(col(prnfail(fail)))
                    print(' ' * len(msg), end=' ')
            if verbose > 1 and (not (prn or prnfail)):
                print('recv:%i  fail:%i' % tuple(map(len, res[:2])))
            if verbose == 1:
                if res[0]:
                    os.write(1, b'*')
                if res[1]:
                    os.write(1, b'.')
            if store:
                ans += res[0]
                unans += res[1]
            end = time.monotonic()
            if end - start < inter:
                time.sleep(inter + start - end)
    except KeyboardInterrupt:
        pass
    if verbose and n > 0:
        print(ct.normal('\nSent %i packets, received %i packets. %3.1f%% hits.' % (n, r, 100.0 * r / n)))
    return (SndRcvList(ans), PacketList(unans))

@conf.commands.register
def srloop(pkts, *args, **kargs):
    if False:
        print('Hello World!')
    '\n    Send a packet at layer 3 in loop and print the answer each time\n    srloop(pkts, [prn], [inter], [count], ...) --> None\n    '
    return __sr_loop(sr, pkts, *args, **kargs)

@conf.commands.register
def srploop(pkts, *args, **kargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Send a packet at layer 2 in loop and print the answer each time\n    srloop(pkts, [prn], [inter], [count], ...) --> None\n    '
    return __sr_loop(srp, pkts, *args, **kargs)

class _FloodGenerator(object):

    def __init__(self, tobesent, maxretries):
        if False:
            while True:
                i = 10
        self.tobesent = tobesent
        self.maxretries = maxretries
        self.stopevent = Event()
        self.iterlen = 0

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        i = 0
        while True:
            i += 1
            j = 0
            if self.maxretries and i >= self.maxretries:
                return
            for p in self.tobesent:
                if self.stopevent.is_set():
                    return
                j += 1
                yield p
            if self.iterlen == 0:
                self.iterlen = j

    @property
    def sent_time(self):
        if False:
            i = 10
            return i + 15
        return cast(Packet, self.tobesent).sent_time

    @sent_time.setter
    def sent_time(self, val):
        if False:
            print('Hello World!')
        cast(Packet, self.tobesent).sent_time = val

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        self.stopevent.set()

def sndrcvflood(pks, pkt, inter=0, maxretries=None, verbose=None, chainCC=False, timeout=None):
    if False:
        i = 10
        return i + 15
    'sndrcv equivalent for flooding.'
    flood_gen = _FloodGenerator(pkt, maxretries)
    return sndrcv(pks, flood_gen, inter=inter, verbose=verbose, chainCC=chainCC, timeout=timeout, _flood=flood_gen)

@conf.commands.register
def srflood(x, promisc=None, filter=None, iface=None, nofilter=None, *args, **kargs):
    if False:
        while True:
            i = 10
    'Flood and receive packets at layer 3\n\n    :param prn:      function applied to packets received\n    :param unique:   only consider packets whose print\n    :param nofilter: put 1 to avoid use of BPF filters\n    :param filter:   provide a BPF filter\n    :param iface:    listen answers only on the given interface\n    '
    (iface, ipv6) = _interface_selection(iface, x)
    s = iface.l3socket(ipv6)(promisc=promisc, filter=filter, iface=iface, nofilter=nofilter)
    r = sndrcvflood(s, x, *args, **kargs)
    s.close()
    return r

@conf.commands.register
def sr1flood(x, promisc=None, filter=None, iface=None, nofilter=0, *args, **kargs):
    if False:
        print('Hello World!')
    'Flood and receive packets at layer 3 and return only the first answer\n\n    :param prn:      function applied to packets received\n    :param verbose:  set verbosity level\n    :param nofilter: put 1 to avoid use of BPF filters\n    :param filter:   provide a BPF filter\n    :param iface:    listen answers only on the given interface\n    '
    (iface, ipv6) = _interface_selection(iface, x)
    s = iface.l3socket(ipv6)(promisc=promisc, filter=filter, nofilter=nofilter, iface=iface)
    (ans, _) = sndrcvflood(s, x, *args, **kargs)
    s.close()
    if len(ans) > 0:
        return cast(Packet, ans[0][1])
    return None

@conf.commands.register
def srpflood(x, promisc=None, filter=None, iface=None, iface_hint=None, nofilter=None, *args, **kargs):
    if False:
        return 10
    'Flood and receive packets at layer 2\n\n    :param prn:      function applied to packets received\n    :param unique:   only consider packets whose print\n    :param nofilter: put 1 to avoid use of BPF filters\n    :param filter:   provide a BPF filter\n    :param iface:    listen answers only on the given interface\n    '
    if iface is None and iface_hint is not None:
        iface = conf.route.route(iface_hint)[0]
    iface = resolve_iface(iface or conf.iface)
    s = iface.l2socket()(promisc=promisc, filter=filter, iface=iface, nofilter=nofilter)
    r = sndrcvflood(s, x, *args, **kargs)
    s.close()
    return r

@conf.commands.register
def srp1flood(x, promisc=None, filter=None, iface=None, nofilter=0, *args, **kargs):
    if False:
        return 10
    'Flood and receive packets at layer 2 and return only the first answer\n\n    :param prn:      function applied to packets received\n    :param verbose:  set verbosity level\n    :param nofilter: put 1 to avoid use of BPF filters\n    :param filter:   provide a BPF filter\n    :param iface:    listen answers only on the given interface\n    '
    iface = resolve_iface(iface or conf.iface)
    s = iface.l2socket()(promisc=promisc, filter=filter, nofilter=nofilter, iface=iface)
    (ans, _) = sndrcvflood(s, x, *args, **kargs)
    s.close()
    if len(ans) > 0:
        return cast(Packet, ans[0][1])
    return None

class AsyncSniffer(object):
    """
    Sniff packets and return a list of packets.

    Args:
        count: number of packets to capture. 0 means infinity.
        store: whether to store sniffed packets or discard them
        prn: function to apply to each packet. If something is returned, it
             is displayed.
             --Ex: prn = lambda x: x.summary()
        session: a session = a flow decoder used to handle stream of packets.
                 --Ex: session=TCPSession
                 See below for more details.
        filter: BPF filter to apply.
        lfilter: Python function applied to each packet to determine if
                 further action may be done.
                 --Ex: lfilter = lambda x: x.haslayer(Padding)
        offline: PCAP file (or list of PCAP files) to read packets from,
                 instead of sniffing them
        quiet:   when set to True, the process stderr is discarded
                 (default: False).
        timeout: stop sniffing after a given time (default: None).
        L2socket: use the provided L2socket (default: use conf.L2listen).
        opened_socket: provide an object (or a list of objects) ready to use
                      .recv() on.
        stop_filter: Python function applied to each packet to determine if
                     we have to stop the capture after this packet.
                     --Ex: stop_filter = lambda x: x.haslayer(TCP)
        iface: interface or list of interfaces (default: None for sniffing
               on all interfaces).
        monitor: use monitor mode. May not be available on all OS
        started_callback: called as soon as the sniffer starts sniffing
                          (default: None).

    The iface, offline and opened_socket parameters can be either an
    element, a list of elements, or a dict object mapping an element to a
    label (see examples below).

    For more information about the session argument, see
    https://scapy.rtfd.io/en/latest/usage.html#advanced-sniffing-sniffing-sessions

    Examples: synchronous
      >>> sniff(filter="arp")
      >>> sniff(filter="tcp",
      ...       session=IPSession,  # defragment on-the-flow
      ...       prn=lambda x: x.summary())
      >>> sniff(lfilter=lambda pkt: ARP in pkt)
      >>> sniff(iface="eth0", prn=Packet.summary)
      >>> sniff(iface=["eth0", "mon0"],
      ...       prn=lambda pkt: "%s: %s" % (pkt.sniffed_on,
      ...                                   pkt.summary()))
      >>> sniff(iface={"eth0": "Ethernet", "mon0": "Wifi"},
      ...       prn=lambda pkt: "%s: %s" % (pkt.sniffed_on,
      ...                                   pkt.summary()))

    Examples: asynchronous
      >>> t = AsyncSniffer(iface="enp0s3")
      >>> t.start()
      >>> time.sleep(1)
      >>> print("nice weather today")
      >>> t.stop()
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.args = args
        self.kwargs = kwargs
        self.running = False
        self.thread = None
        self.results = None

    def _setup_thread(self):
        if False:
            for i in range(10):
                print('nop')
        self.thread = Thread(target=self._run, args=self.args, kwargs=self.kwargs, name='AsyncSniffer')
        self.thread.daemon = True

    def _run(self, count=0, store=True, offline=None, quiet=False, prn=None, lfilter=None, L2socket=None, timeout=None, opened_socket=None, stop_filter=None, iface=None, started_callback=None, session=None, **karg):
        if False:
            print('Hello World!')
        self.running = True
        self.count = 0
        lst = []
        if not isinstance(session, DefaultSession):
            session = session or DefaultSession
            session = session()
        sniff_sockets = {}
        if opened_socket is not None:
            if isinstance(opened_socket, list):
                sniff_sockets.update(((s, 'socket%d' % i) for (i, s) in enumerate(opened_socket)))
            elif isinstance(opened_socket, dict):
                sniff_sockets.update(((s, label) for (s, label) in opened_socket.items()))
            else:
                sniff_sockets[opened_socket] = 'socket0'
        if offline is not None:
            flt = karg.get('filter')
            if isinstance(offline, str):
                offline = [offline]
            if isinstance(offline, list) and all((isinstance(elt, str) for elt in offline)):
                sniff_sockets.update(((PcapReader(fname if flt is None else tcpdump(fname, args=['-w', '-'], flt=flt, getfd=True, quiet=quiet)), fname) for fname in offline))
            elif isinstance(offline, dict):
                sniff_sockets.update(((PcapReader(fname if flt is None else tcpdump(fname, args=['-w', '-'], flt=flt, getfd=True, quiet=quiet)), label) for (fname, label) in offline.items()))
            elif isinstance(offline, (Packet, PacketList, list)):
                offline = IterSocket(offline)
                sniff_sockets[offline if flt is None else PcapReader(tcpdump(offline, args=['-w', '-'], flt=flt, getfd=True, quiet=quiet))] = offline
            else:
                sniff_sockets[PcapReader(offline if flt is None else tcpdump(offline, args=['-w', '-'], flt=flt, getfd=True, quiet=quiet))] = offline
        if not sniff_sockets or iface is not None:
            _RL2 = lambda i: L2socket or resolve_iface(i).l2listen()
            if isinstance(iface, list):
                sniff_sockets.update(((_RL2(ifname)(type=ETH_P_ALL, iface=ifname, **karg), ifname) for ifname in iface))
            elif isinstance(iface, dict):
                sniff_sockets.update(((_RL2(ifname)(type=ETH_P_ALL, iface=ifname, **karg), iflabel) for (ifname, iflabel) in iface.items()))
            else:
                iface = iface or conf.iface
                sniff_sockets[_RL2(iface)(type=ETH_P_ALL, iface=iface, **karg)] = iface
        _main_socket = next(iter(sniff_sockets))
        select_func = _main_socket.select
        nonblocking_socket = getattr(_main_socket, 'nonblocking_socket', False)
        if not all((select_func == sock.select for sock in sniff_sockets)):
            warning('Warning: inconsistent socket types ! The used select function will be the one of the first socket')
        close_pipe = None
        if not nonblocking_socket:
            from scapy.automaton import ObjectPipe
            close_pipe = ObjectPipe[None]()
            sniff_sockets[close_pipe] = 'control_socket'

            def stop_cb():
                if False:
                    print('Hello World!')
                if self.running and close_pipe:
                    close_pipe.send(None)
                self.continue_sniff = False
            self.stop_cb = stop_cb
        else:

            def stop_cb():
                if False:
                    for i in range(10):
                        print('nop')
                self.continue_sniff = False
            self.stop_cb = stop_cb
        try:
            if started_callback:
                started_callback()
            self.continue_sniff = True
            if timeout is not None:
                stoptime = time.monotonic() + timeout
            remain = None
            while sniff_sockets and self.continue_sniff:
                if timeout is not None:
                    remain = stoptime - time.monotonic()
                    if remain <= 0:
                        break
                sockets = select_func(list(sniff_sockets.keys()), remain)
                dead_sockets = []
                for s in sockets:
                    if s is close_pipe:
                        break
                    try:
                        packets = session.recv(s)
                        for p in packets:
                            if lfilter and (not lfilter(p)):
                                continue
                            p.sniffed_on = sniff_sockets[s]
                            self.count += 1
                            if store:
                                lst.append(p)
                            if prn:
                                result = prn(p)
                                if result is not None:
                                    print(result)
                            if stop_filter and stop_filter(p) or 0 < count <= self.count:
                                self.continue_sniff = False
                                break
                    except EOFError:
                        try:
                            s.close()
                        except Exception:
                            pass
                        dead_sockets.append(s)
                        continue
                    except Exception as ex:
                        msg = ' It was closed.'
                        try:
                            s.close()
                        except Exception as ex2:
                            msg = " close() failed with '%s'" % ex2
                        warning("Socket %s failed with '%s'." % (s, ex) + msg)
                        dead_sockets.append(s)
                        if conf.debug_dissector >= 2:
                            raise
                        continue
                for s in dead_sockets:
                    del sniff_sockets[s]
                    if len(sniff_sockets) == 1 and close_pipe in sniff_sockets:
                        del sniff_sockets[close_pipe]
        except KeyboardInterrupt:
            pass
        self.running = False
        if opened_socket is None:
            for s in sniff_sockets:
                s.close()
        elif close_pipe:
            close_pipe.close()
        self.results = PacketList(lst, 'Sniffed')

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Starts AsyncSniffer in async mode'
        self._setup_thread()
        if self.thread:
            self.thread.start()

    def stop(self, join=True):
        if False:
            print('Hello World!')
        'Stops AsyncSniffer if not in async mode'
        if self.running:
            try:
                self.stop_cb()
            except AttributeError:
                raise Scapy_Exception('Unsupported (offline or unsupported socket)')
            if join:
                self.join()
                return self.results
            return None
        else:
            raise Scapy_Exception('Not running ! (check .running attr)')

    def join(self, *args, **kwargs):
        if False:
            return 10
        if self.thread:
            self.thread.join(*args, **kwargs)

@conf.commands.register
def sniff(*args, **kwargs):
    if False:
        while True:
            i = 10
    sniffer = AsyncSniffer()
    sniffer._run(*args, **kwargs)
    return cast(PacketList, sniffer.results)
sniff.__doc__ = AsyncSniffer.__doc__

@conf.commands.register
def bridge_and_sniff(if1, if2, xfrm12=None, xfrm21=None, prn=None, L2socket=None, *args, **kargs):
    if False:
        return 10
    'Forward traffic between interfaces if1 and if2, sniff and return\n    the exchanged packets.\n\n    :param if1: the interfaces to use (interface names or opened sockets).\n    :param if2:\n    :param xfrm12: a function to call when forwarding a packet from if1 to\n        if2. If it returns True, the packet is forwarded as it. If it\n        returns False or None, the packet is discarded. If it returns a\n        packet, this packet is forwarded instead of the original packet\n        one.\n    :param xfrm21: same as xfrm12 for packets forwarded from if2 to if1.\n\n    The other arguments are the same than for the function sniff(),\n    except for offline, opened_socket and iface that are ignored.\n    See help(sniff) for more.\n    '
    for arg in ['opened_socket', 'offline', 'iface']:
        if arg in kargs:
            log_runtime.warning('Argument %s cannot be used in bridge_and_sniff() -- ignoring it.', arg)
            del kargs[arg]

    def _init_socket(iface, count, L2socket=L2socket):
        if False:
            return 10
        if isinstance(iface, SuperSocket):
            return (iface, 'iface%d' % count)
        else:
            if not L2socket:
                iface = resolve_iface(iface or conf.iface)
                L2socket = iface.l2socket()
            return (L2socket(iface=iface), iface)
    (sckt1, if1) = _init_socket(if1, 1)
    (sckt2, if2) = _init_socket(if2, 2)
    peers = {if1: sckt2, if2: sckt1}
    xfrms = {}
    if xfrm12 is not None:
        xfrms[if1] = xfrm12
    if xfrm21 is not None:
        xfrms[if2] = xfrm21

    def prn_send(pkt):
        if False:
            return 10
        try:
            sendsock = peers[pkt.sniffed_on or '']
        except KeyError:
            return
        if pkt.sniffed_on in xfrms:
            try:
                _newpkt = xfrms[pkt.sniffed_on](pkt)
            except Exception:
                log_runtime.warning('Exception in transformation function for packet [%s] received on %s -- dropping', pkt.summary(), pkt.sniffed_on, exc_info=True)
                return
            else:
                if isinstance(_newpkt, bool):
                    if not _newpkt:
                        return
                    newpkt = pkt
                else:
                    newpkt = _newpkt
        else:
            newpkt = pkt
        try:
            sendsock.send(newpkt)
        except Exception:
            log_runtime.warning('Cannot forward packet [%s] received on %s', pkt.summary(), pkt.sniffed_on, exc_info=True)
    if prn is None:
        prn = prn_send
    else:
        prn_orig = prn

        def prn(pkt):
            if False:
                for i in range(10):
                    print('nop')
            prn_send(pkt)
            return prn_orig(pkt)
    return sniff(*args, opened_socket={sckt1: if1, sckt2: if2}, prn=prn, **kargs)

@conf.commands.register
def tshark(*args, **kargs):
    if False:
        while True:
            i = 10
    'Sniff packets and print them calling pkt.summary().\n    This tries to replicate what text-wireshark (tshark) would look like'
    if 'iface' in kargs:
        iface = kargs.get('iface')
    elif 'opened_socket' in kargs:
        iface = cast(SuperSocket, kargs.get('opened_socket')).iface
    else:
        iface = conf.iface
    print("Capturing on '%s'" % iface)
    i = [0]

    def _cb(pkt):
        if False:
            while True:
                i = 10
        print('%5d\t%s' % (i[0], pkt.summary()))
        i[0] += 1
    sniff(*args, prn=_cb, store=False, **kargs)
    print('\n%d packet%s captured' % (i[0], 's' if i[0] > 1 else ''))