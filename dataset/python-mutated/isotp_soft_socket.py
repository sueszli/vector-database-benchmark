import logging
import struct
import time
import traceback
import heapq
import socket
from threading import Thread, Event, RLock
from bisect import bisect_left
from scapy.packet import Packet
from scapy.layers.can import CAN
from scapy.error import Scapy_Exception
from scapy.supersocket import SuperSocket
from scapy.config import conf
from scapy.consts import LINUX
from scapy.utils import EDecimal
from scapy.automaton import ObjectPipe, select_objects
from scapy.contrib.isotp.isotp_packet import ISOTP, CAN_MAX_DLEN, N_PCI_SF, N_PCI_CF, N_PCI_FC, N_PCI_FF, ISOTP_MAX_DLEN, ISOTP_MAX_DLEN_2015, CAN_FD_MAX_DLEN
from typing import Optional, Union, List, Tuple, Any, Type, cast, Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from scapy.contrib.cansocket import CANSocket
log_isotp = logging.getLogger('scapy.contrib.isotp')
ISOTP_IDLE = 0
ISOTP_WAIT_FIRST_FC = 1
ISOTP_WAIT_FC = 2
ISOTP_WAIT_DATA = 3
ISOTP_SENDING = 4
ISOTP_FC_CTS = 0
ISOTP_FC_WT = 1
ISOTP_FC_OVFLW = 2

class ISOTPSoftSocket(SuperSocket):
    """
    This class is a wrapper around the ISOTPSocketImplementation, for the
    reasons described below.

    The ISOTPSoftSocket aims to be fully compatible with the Linux ISOTP
    sockets provided by the can-isotp kernel module, while being usable on any
    operating system.
    Therefore, this socket needs to be able to respond to an incoming FF frame
    with a FC frame even before the recv() method is called.
    A thread is needed for receiving CAN frames in the background, and since
    the lower layer CAN implementation is not guaranteed to have a functioning
    POSIX select(), each ISOTP socket needs its own CAN receiver thread.
    SuperSocket automatically calls the close() method when the GC destroys an
    ISOTPSoftSocket. However, note that if any thread holds a reference to
    an ISOTPSoftSocket object, it will not be collected by the GC.

    The implementation of the ISOTP protocol, along with the necessary
    thread, are stored in the ISOTPSocketImplementation class, and therefore:

    * There no reference from ISOTPSocketImplementation to ISOTPSoftSocket
    * ISOTPSoftSocket can be normally garbage collected
    * Upon destruction, ISOTPSoftSocket.close() will be called
    * ISOTPSoftSocket.close() will call ISOTPSocketImplementation.close()
    * RX background thread can be stopped by the garbage collector

    Initialize an ISOTPSoftSocket using the provided underlying can socket.

    Example (with NativeCANSocket underneath):
        >>> conf.contribs['ISOTP'] = {'use-can-isotp-kernel-module': False}
        >>> load_contrib('isotp')
        >>> with ISOTPSocket("can0", tx_id=0x641, rx_id=0x241) as sock:
        >>>     sock.send(...)

    Example (with PythonCANSocket underneath):
        >>> conf.contribs['ISOTP'] = {'use-can-isotp-kernel-module': False}
        >>> conf.contribs['CANSocket'] = {'use-python-can': True}
        >>> load_contrib('isotp')
        >>> with ISOTPSocket(CANSocket(bustype='socketcan', channel="can0"), tx_id=0x641, rx_id=0x241) as sock:
        >>>     sock.send(...)

    :param can_socket: a CANSocket instance, preferably filtering only can
                       frames with identifier equal to rx_id
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
    :param basecls: base class of the packets emitted by this socket
    :param fd: enables the CanFD support for this socket
    """

    def __init__(self, can_socket=None, tx_id=0, rx_id=0, ext_address=None, rx_ext_address=None, bs=0, stmin=0, padding=False, listen_only=False, basecls=ISOTP, fd=False):
        if False:
            for i in range(10):
                print('nop')
        if LINUX and isinstance(can_socket, str):
            from scapy.contrib.cansocket_native import NativeCANSocket
            can_socket = NativeCANSocket(can_socket, fd=fd)
        elif isinstance(can_socket, str):
            raise Scapy_Exception('Provide a CANSocket object instead')
        self.ext_address = ext_address
        self.rx_ext_address = rx_ext_address or ext_address
        self.tx_id = tx_id
        self.rx_id = rx_id
        self.fd = fd
        impl = ISOTPSocketImplementation(can_socket, tx_id=self.tx_id, rx_id=self.rx_id, padding=padding, ext_address=self.ext_address, rx_ext_address=self.rx_ext_address, bs=bs, stmin=stmin, listen_only=listen_only, fd=fd)
        self.ins = cast(socket.socket, impl)
        self.outs = cast(socket.socket, impl)
        self.impl = impl
        self.basecls = basecls
        if basecls is None:
            log_isotp.warning('Provide a basecls ')

    def close(self):
        if False:
            return 10
        if not self.closed:
            self.impl.close()
            self.closed = True

    def failure_analysis(self):
        if False:
            i = 10
            return i + 15
        self.impl.failure_analysis()

    def recv_raw(self, x=65535):
        if False:
            return 10
        "Receive a complete ISOTP message, blocking until a message is\n        received or the specified timeout is reached.\n        If self.timeout is 0, then this function doesn't block and returns the\n        first frame in the receive buffer or None if there isn't any."
        if not self.closed:
            tup = self.impl.recv()
            if tup is not None:
                return (self.basecls, tup[0], float(tup[1]))
        return (self.basecls, None, None)

    def recv(self, x=65535, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        msg = super(ISOTPSoftSocket, self).recv(x, **kwargs)
        if msg is None:
            return None
        if hasattr(msg, 'tx_id'):
            msg.tx_id = self.tx_id
        if hasattr(msg, 'rx_id'):
            msg.rx_id = self.rx_id
        if hasattr(msg, 'ext_address'):
            msg.ext_address = self.ext_address
        if hasattr(msg, 'rx_ext_address'):
            msg.rx_ext_address = self.rx_ext_address
        return msg

    @staticmethod
    def select(sockets, remain=None):
        if False:
            i = 10
            return i + 15
        'This function is called during sendrecv() routine to wait for\n        sockets to be ready to receive\n        '
        obj_pipes = [x.impl.rx_queue for x in sockets if isinstance(x, ISOTPSoftSocket) and (not x.closed)]
        ready_pipes = select_objects(obj_pipes, remain)
        return [x for x in sockets if isinstance(x, ISOTPSoftSocket) and (not x.closed) and (x.impl.rx_queue in ready_pipes)]

class TimeoutScheduler:
    """A timeout scheduler which uses a single thread for all timeouts, unlike
    python's own Timer objects which use a thread each."""
    GRACE = 0.1
    _mutex = RLock()
    _event = Event()
    _thread = None
    _handles = []
    logger = logging.getLogger('scapy.contrib.automotive.timeout_scheduler')

    @classmethod
    def schedule(cls, timeout, callback):
        if False:
            for i in range(10):
                print('nop')
        'Schedules the execution of a timeout.\n\n        The function `callback` will be called in `timeout` seconds.\n\n        Returns a handle that can be used to remove the timeout.'
        when = cls._time() + timeout
        handle = cls.Handle(when, callback)
        with cls._mutex:
            heapq.heappush(cls._handles, handle)
            must_interrupt = cls._handles[0] == handle
            if cls._thread is None:
                t = Thread(target=cls._task, name='TimeoutScheduler._task')
                must_interrupt = False
                cls._thread = t
                cls._event.clear()
                t.start()
        if must_interrupt:
            cls._event.set()
            time.sleep(0)
        return handle

    @classmethod
    def cancel(cls, handle):
        if False:
            i = 10
            return i + 15
        'Provided its handle, cancels the execution of a timeout.'
        with cls._mutex:
            if handle in cls._handles:
                handle._cb = None
                cls._handles.remove(handle)
                heapq.heapify(cls._handles)
                if len(cls._handles) == 0:
                    cls._event.set()
            else:
                raise Scapy_Exception('Handle not found')

    @classmethod
    def clear(cls):
        if False:
            for i in range(10):
                print('nop')
        'Cancels the execution of all timeouts.'
        with cls._mutex:
            cls._handles = []
        cls._event.set()

    @classmethod
    def _peek_next(cls):
        if False:
            i = 10
            return i + 15
        'Returns the next timeout to execute, or `None` if list is empty,\n        without modifying the list'
        with cls._mutex:
            return cls._handles[0] if cls._handles else None

    @classmethod
    def _wait(cls, handle):
        if False:
            i = 10
            return i + 15
        'Waits until it is time to execute the provided handle, or until\n        another thread calls _event.set()'
        now = cls._time()
        if handle is None:
            to_wait = cls.GRACE
        else:
            to_wait = handle._when - now
        if to_wait > 0:
            cls.logger.debug('Thread going to sleep @ %f ' + 'for %fs', now, to_wait)
            interrupted = cls._event.wait(to_wait)
            new = cls._time()
            cls.logger.debug('Thread awake @ %f, slept for' + ' %f, interrupted=%d', new, new - now, interrupted)
        cls._event.clear()

    @classmethod
    def _task(cls):
        if False:
            print('Hello World!')
        'Executed in a background thread, this thread will automatically\n        start when the first timeout is added and stop when the last timeout\n        is removed or executed.'
        cls.logger.debug('Thread spawning @ %f', cls._time())
        time_empty = None
        try:
            while 1:
                handle = cls._peek_next()
                if handle is None:
                    now = cls._time()
                    if time_empty is None:
                        time_empty = now
                    if cls.GRACE < now - time_empty:
                        return
                else:
                    time_empty = None
                cls._wait(handle)
                cls._poll()
        finally:
            cls.logger.debug('Thread died @ %f', cls._time())
            cls._thread = None

    @classmethod
    def _poll(cls):
        if False:
            while True:
                i = 10
        'Execute all the callbacks that were due until now'
        while 1:
            with cls._mutex:
                now = cls._time()
                if len(cls._handles) == 0 or cls._handles[0]._when > now:
                    return
                handle = heapq.heappop(cls._handles)
                callback = None
                if handle is not None:
                    callback = handle._cb
                    handle._cb = True
            if callable(callback):
                try:
                    callback()
                except Exception:
                    traceback.print_exc()

    @staticmethod
    def _time():
        if False:
            return 10
        return time.monotonic()

    class Handle:
        """Handle for a timeout, consisting of a callback and a time when it
        should be executed."""
        __slots__ = ['_when', '_cb']

        def __init__(self, when, cb):
            if False:
                print('Hello World!')
            self._when = when
            self._cb = cb

        def cancel(self):
            if False:
                while True:
                    i = 10
            'Cancels this timeout, preventing it from executing its\n            callback'
            if self._cb is None:
                raise Scapy_Exception('cancel() called on previous canceled Handle')
            else:
                with TimeoutScheduler._mutex:
                    if isinstance(self._cb, bool):
                        return False
                    else:
                        self._cb = None
                        TimeoutScheduler.cancel(self)
                        return True

        def __lt__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(other, TimeoutScheduler.Handle):
                raise TypeError()
            return self._when < other._when

        def __le__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(other, TimeoutScheduler.Handle):
                raise TypeError()
            return self._when <= other._when

        def __gt__(self, other):
            if False:
                i = 10
                return i + 15
            if not isinstance(other, TimeoutScheduler.Handle):
                raise TypeError()
            return self._when > other._when

        def __ge__(self, other):
            if False:
                return 10
            if not isinstance(other, TimeoutScheduler.Handle):
                raise TypeError()
            return self._when >= other._when

class ISOTPSocketImplementation:
    """
    Implementation of an ISOTP "state machine".

    Most of the ISOTP logic was taken from
    https://github.com/hartkopp/can-isotp/blob/master/net/can/isotp.c

    This class is separated from ISOTPSoftSocket to make sure the background
    thread can't hold a reference to ISOTPSoftSocket, allowing it to be
    collected by the GC.

    :param can_socket: a CANSocket instance, preferably filtering only can
                       frames with identifier equal to rx_id
    :param tx_id: the CAN identifier of the sent CAN frames
    :param rx_id: the CAN identifier of the received CAN frames
    :param padding: If True, pads sending packets with 0x00 which not
                    count to the payload.
                    Does not affect receiving packets.
    :param ext_address: Extended Address byte to be added at the
            beginning of every CAN frame _sent_ by this object. Can be None
            in order to disable extended addressing on sent frames.
    :param rx_ext_address: Extended Address byte expected to be found at
            the beginning of every CAN frame _received_ by this object. Can
            be None in order to disable extended addressing on received
            frames.
    :param bs: Block Size byte to be included in every Control
            Flow Frame sent by this object. The default value of 0 means
            that all the data will be received in a single block.
    :param stmin: Time Minimum Separation byte to be
            included in every Control Flow Frame sent by this object. The
            default value of 0 indicates that the peer will not wait any
            time between sending frames.
    :param listen_only: Disables send of flow control frames
    """

    def __init__(self, can_socket, tx_id, rx_id, padding=False, ext_address=None, rx_ext_address=None, bs=0, stmin=0, listen_only=False, fd=False):
        if False:
            i = 10
            return i + 15
        self.can_socket = can_socket
        self.rx_id = rx_id
        self.tx_id = tx_id
        self.padding = padding
        self.fc_timeout = 1
        self.cf_timeout = 1
        self.fd = fd
        self.max_dlen = CAN_FD_MAX_DLEN if fd else CAN_MAX_DLEN
        self.filter_warning_emitted = False
        self.closed = False
        self.rx_ext_address = rx_ext_address
        self.ea_hdr = b''
        if ext_address is not None:
            self.ea_hdr = struct.pack('B', ext_address)
        self.listen_only = listen_only
        self.rxfc_bs = bs
        self.rxfc_stmin = stmin
        self.rx_queue = ObjectPipe[Tuple[bytes, Union[float, EDecimal]]]()
        self.rx_len = -1
        self.rx_buf = None
        self.rx_sn = 0
        self.rx_bs = 0
        self.rx_idx = 0
        self.rx_ts = 0.0
        self.rx_state = ISOTP_IDLE
        self.tx_queue = ObjectPipe[bytes]()
        self.txfc_bs = 0
        self.txfc_stmin = 0
        self.tx_gap = 0.0
        self.tx_buf = None
        self.tx_sn = 0
        self.tx_bs = 0
        self.tx_idx = 0
        self.rx_ll_dl = 0
        self.tx_state = ISOTP_IDLE
        self.rx_tx_poll_rate = 0.005
        self.tx_timeout_handle = None
        self.rx_timeout_handle = None
        self.rx_handle = TimeoutScheduler.schedule(self.rx_tx_poll_rate, self.can_recv)
        self.tx_handle = TimeoutScheduler.schedule(self.rx_tx_poll_rate, self._send)
        self.last_rx_call = 0.0

    def failure_analysis(self):
        if False:
            for i in range(10):
                print('nop')
        log_isotp.debug('Failure analysis')
        log_isotp.debug('Last_rx_call: %s', str(self.last_rx_call))
        log_isotp.debug('self.rx_handle: %s', str(self.rx_handle))
        log_isotp.debug('self.rx_handle._cb: %s', str(self.rx_handle._cb))
        log_isotp.debug('self.rx_handle._when: %s', str(self.rx_handle._when))
        log_isotp.debug('Now: %s', TimeoutScheduler._time())

    def __del__(self):
        if False:
            while True:
                i = 10
        self.close()

    def can_send(self, load):
        if False:
            print('Hello World!')

        def _get_padding_size(pl_size):
            if False:
                for i in range(10):
                    print('nop')
            if not self.fd:
                return CAN_MAX_DLEN
            else:
                fd_accepted_sizes = [0, 8, 12, 16, 20, 24, 32, 48, 64]
                pos = bisect_left(fd_accepted_sizes, pl_size)
                if pos == 0:
                    return fd_accepted_sizes[0]
                if pos == len(fd_accepted_sizes):
                    return fd_accepted_sizes[-1]
                return fd_accepted_sizes[pos]
        if self.padding:
            load += b'\xcc' * (_get_padding_size(len(load)) - len(load))
        if self.tx_id is None or self.tx_id <= 2047:
            self.can_socket.send(CAN(identifier=self.tx_id, data=load))
        else:
            self.can_socket.send(CAN(identifier=self.tx_id, flags='extended', data=load))

    def can_recv(self):
        if False:
            return 10
        self.last_rx_call = TimeoutScheduler._time()
        if self.can_socket.select([self.can_socket], 0):
            pkt = self.can_socket.recv()
            if pkt:
                self.on_can_recv(pkt)
        if not self.closed and (not self.can_socket.closed):
            if self.can_socket.select([self.can_socket], 0):
                poll_time = 0.0
            else:
                poll_time = self.rx_tx_poll_rate
            self.rx_handle = TimeoutScheduler.schedule(poll_time, self.can_recv)
        else:
            try:
                self.rx_handle.cancel()
            except Scapy_Exception:
                pass

    def on_can_recv(self, p):
        if False:
            print('Hello World!')
        if p.identifier != self.rx_id:
            if not self.filter_warning_emitted and conf.verb >= 2:
                log_isotp.warning('You should put a filter for identifier=%x on your CAN socket', self.rx_id)
                self.filter_warning_emitted = True
        else:
            self.on_recv(p)

    def close(self):
        if False:
            return 10
        try:
            if select_objects([self.tx_queue], 0):
                log_isotp.warning('TX queue not empty')
                time.sleep(0.1)
        except OSError:
            pass
        try:
            if select_objects([self.rx_queue], 0):
                log_isotp.warning('RX queue not empty')
        except OSError:
            pass
        self.closed = True
        try:
            self.rx_handle.cancel()
        except Scapy_Exception:
            pass
        try:
            self.tx_handle.cancel()
        except Scapy_Exception:
            pass

    def _rx_timer_handler(self):
        if False:
            return 10
        'Method called every time the rx_timer times out, due to the peer not\n        sending a consecutive frame within the expected time window'
        if self.rx_state == ISOTP_WAIT_DATA:
            self.rx_state = ISOTP_IDLE
            if conf.verb > 2:
                log_isotp.warning('RX state was reset due to timeout')

    def _tx_timer_handler(self):
        if False:
            i = 10
            return i + 15
        'Method called every time the tx_timer times out, which can happen in\n        two situations: either a Flow Control frame was not received in time,\n        or the Separation Time Min is expired and a new frame must be sent.'
        if self.tx_state == ISOTP_WAIT_FC or self.tx_state == ISOTP_WAIT_FIRST_FC:
            self.tx_state = ISOTP_IDLE
            log_isotp.warning('TX state was reset due to timeout')
            return
        elif self.tx_state == ISOTP_SENDING:
            src_off = len(self.ea_hdr)
            max_bytes = self.max_dlen - 1 - src_off
            if self.tx_buf is None:
                self.tx_state = ISOTP_IDLE
                log_isotp.warning('TX buffer is not filled')
                return
            while 1:
                load = self.ea_hdr
                load += struct.pack('B', N_PCI_CF + self.tx_sn)
                load += self.tx_buf[self.tx_idx:self.tx_idx + max_bytes]
                self.can_send(load)
                self.tx_sn = (self.tx_sn + 1) % 16
                self.tx_bs += 1
                self.tx_idx += max_bytes
                if len(self.tx_buf) <= self.tx_idx:
                    self.tx_state = ISOTP_IDLE
                    return
                if self.txfc_bs != 0 and self.tx_bs >= self.txfc_bs:
                    self.tx_state = ISOTP_WAIT_FC
                    self.tx_timeout_handle = TimeoutScheduler.schedule(self.fc_timeout, self._tx_timer_handler)
                    return
                if self.tx_gap == 0:
                    continue
                else:
                    self.tx_timeout_handle = TimeoutScheduler.schedule(self.tx_gap, self._tx_timer_handler)
                    return

    def on_recv(self, cf):
        if False:
            while True:
                i = 10
        'Function that must be called every time a CAN frame is received, to\n        advance the state machine.'
        data = bytes(cf.data)
        if len(data) < 2:
            return
        ae = 0
        if self.rx_ext_address is not None:
            ae = 1
            if len(data) < 3:
                return
            if data[0] != self.rx_ext_address:
                return
        n_pci = data[ae] & 240
        if n_pci == N_PCI_FC:
            self._recv_fc(data[ae:])
        elif n_pci == N_PCI_SF:
            self._recv_sf(data[ae:], cf.time)
        elif n_pci == N_PCI_FF:
            self._recv_ff(data[ae:], cf.time)
        elif n_pci == N_PCI_CF:
            self._recv_cf(data[ae:])

    def _recv_fc(self, data):
        if False:
            return 10
        "Process a received 'Flow Control' frame"
        log_isotp.debug('Processing FC')
        if self.tx_state != ISOTP_WAIT_FC and self.tx_state != ISOTP_WAIT_FIRST_FC:
            return
        if self.tx_timeout_handle is not None:
            self.tx_timeout_handle.cancel()
            self.tx_timeout_handle = None
        if len(data) < 3:
            self.tx_state = ISOTP_IDLE
            log_isotp.warning('CF frame discarded because it was too short')
            return
        if self.tx_state == ISOTP_WAIT_FIRST_FC:
            self.txfc_bs = data[1]
            self.txfc_stmin = data[2]
        if self.txfc_stmin > 127 and (self.txfc_stmin < 241 or self.txfc_stmin > 249):
            self.txfc_stmin = 127
        if data[2] <= 127:
            self.tx_gap = data[2] / 1000
        elif 241 <= data[2] <= 249:
            self.tx_gap = (data[2] & 15) / 10000
        else:
            self.tx_gap = 0.0
        self.tx_state = ISOTP_WAIT_FC
        isotp_fc = data[0] & 15
        if isotp_fc == ISOTP_FC_CTS:
            self.tx_bs = 0
            self.tx_state = ISOTP_SENDING
            self.tx_timeout_handle = TimeoutScheduler.schedule(self.tx_gap, self._tx_timer_handler)
        elif isotp_fc == ISOTP_FC_WT:
            self.tx_state = ISOTP_WAIT_FC
            self.tx_timeout_handle = TimeoutScheduler.schedule(self.fc_timeout, self._tx_timer_handler)
        elif isotp_fc == ISOTP_FC_OVFLW:
            self.tx_state = ISOTP_IDLE
            log_isotp.warning('Overflow happened at the receiver side')
            return
        else:
            self.tx_state = ISOTP_IDLE
            log_isotp.warning('Unknown FC frame type')
            return

    def _recv_sf(self, data, ts):
        if False:
            for i in range(10):
                print('nop')
        "Process a received 'Single Frame' frame"
        log_isotp.debug('Processing SF')
        if self.rx_timeout_handle is not None:
            self.rx_timeout_handle.cancel()
            self.rx_timeout_handle = None
        if self.rx_state != ISOTP_IDLE:
            if conf.verb > 2:
                log_isotp.warning('RX state was reset because single frame was received')
            self.rx_state = ISOTP_IDLE
        length = data[0] & 15
        is_fd_frame = self.fd and length == 0 and (len(data) >= 2)
        if is_fd_frame:
            length = data[1]
        if len(data) - 1 < length:
            return
        msg = None
        if is_fd_frame:
            msg = data[2:2 + length]
        else:
            msg = data[1:1 + length]
        self.rx_queue.send((msg, ts))

    def _recv_ff(self, data, ts):
        if False:
            print('Hello World!')
        "Process a received 'First Frame' frame"
        log_isotp.debug('Processing FF')
        if self.rx_timeout_handle is not None:
            self.rx_timeout_handle.cancel()
            self.rx_timeout_handle = None
        if self.rx_state != ISOTP_IDLE:
            if conf.verb > 2:
                log_isotp.warning('RX state was reset because first frame was received')
            self.rx_state = ISOTP_IDLE
        if len(data) < 7:
            return
        self.rx_ll_dl = len(data)
        self.rx_len = (data[0] & 15) * 256 + data[1]
        ff_pci_sz = 2
        if self.rx_len == 0:
            self.rx_len = data[2] << 24
            self.rx_len += data[3] << 16
            self.rx_len += data[4] << 8
            self.rx_len += data[5]
            ff_pci_sz = 6
        data_bytes = data[ff_pci_sz:]
        self.rx_idx = len(data_bytes)
        self.rx_buf = data_bytes
        self.rx_ts = ts
        self.rx_sn = 1
        self.rx_state = ISOTP_WAIT_DATA
        if not self.listen_only:
            load = self.ea_hdr
            load += struct.pack('BBB', N_PCI_FC, self.rxfc_bs, self.rxfc_stmin)
            self.can_send(load)
        self.rx_bs = 0
        self.rx_timeout_handle = TimeoutScheduler.schedule(self.cf_timeout, self._rx_timer_handler)

    def _recv_cf(self, data):
        if False:
            print('Hello World!')
        "Process a received 'Consecutive Frame' frame"
        log_isotp.debug('Processing CF')
        if self.rx_state != ISOTP_WAIT_DATA:
            return
        if self.rx_timeout_handle is not None:
            self.rx_timeout_handle.cancel()
            self.rx_timeout_handle = None
        if len(data) > self.rx_ll_dl:
            return
        if len(data) < self.rx_ll_dl:
            if self.rx_len - self.rx_idx > self.rx_ll_dl:
                if conf.verb > 2:
                    log_isotp.warning('Received a CF with insufficient length')
                return
        if data[0] & 15 != self.rx_sn:
            if conf.verb > 2:
                log_isotp.warning('RX state was reset because wrong sequence number was received')
            self.rx_state = ISOTP_IDLE
            return
        if self.rx_buf is None:
            if conf.verb > 2:
                log_isotp.warning('rx_buf not filled with data!')
            self.rx_state = ISOTP_IDLE
            return
        self.rx_sn = (self.rx_sn + 1) % 16
        self.rx_buf += data[1:]
        self.rx_idx = len(self.rx_buf)
        if self.rx_idx >= self.rx_len:
            self.rx_buf = self.rx_buf[0:self.rx_len]
            self.rx_state = ISOTP_IDLE
            self.rx_queue.send((self.rx_buf, self.rx_ts))
            self.rx_buf = None
            return
        if self.rxfc_bs != 0:
            self.rx_bs += 1
            if self.rx_bs >= self.rxfc_bs and (not self.listen_only):
                load = self.ea_hdr
                load += struct.pack('BBB', N_PCI_FC, self.rxfc_bs, self.rxfc_stmin)
                self.rx_bs = 0
                self.can_send(load)
        log_isotp.debug('Wait for another CF')
        self.rx_timeout_handle = TimeoutScheduler.schedule(self.cf_timeout, self._rx_timer_handler)

    def begin_send(self, x):
        if False:
            while True:
                i = 10
        'Begins sending an ISOTP message. This method does not block.'
        if self.tx_state != ISOTP_IDLE:
            log_isotp.warning('Socket is already sending, retry later')
            return
        self.tx_state = ISOTP_SENDING
        length = len(x)
        if length > ISOTP_MAX_DLEN_2015:
            log_isotp.warning('Too much data for ISOTP message')
        sf_size_check = self.max_dlen - 1
        if len(self.ea_hdr) + length + int(self.fd) <= sf_size_check:
            data = self.ea_hdr
            if not self.fd or length <= 7:
                data += struct.pack('B', length)
            else:
                data += struct.pack('BB', 0, length)
            data += x
            self.tx_state = ISOTP_IDLE
            self.can_send(data)
            return
        data = self.ea_hdr
        if length > ISOTP_MAX_DLEN:
            data += struct.pack('>HI', 4096, length)
        else:
            data += struct.pack('>H', 4096 | length)
        load = x[0:self.max_dlen - len(data)]
        data += load
        self.can_send(data)
        self.tx_buf = x
        self.tx_sn = 1
        self.tx_bs = 0
        self.tx_idx = len(load)
        self.tx_state = ISOTP_WAIT_FIRST_FC
        self.tx_timeout_handle = TimeoutScheduler.schedule(self.fc_timeout, self._tx_timer_handler)

    def _send(self):
        if False:
            print('Hello World!')
        if self.tx_state == ISOTP_IDLE:
            if select_objects([self.tx_queue], 0):
                pkt = self.tx_queue.recv()
                if pkt:
                    self.begin_send(pkt)
        if not self.closed:
            self.tx_handle = TimeoutScheduler.schedule(self.rx_tx_poll_rate, self._send)
        else:
            try:
                self.tx_handle.cancel()
            except Scapy_Exception:
                pass

    def send(self, p):
        if False:
            while True:
                i = 10
        'Send an ISOTP frame and block until the message is sent or an error\n        happens.'
        self.tx_queue.send(p)

    def recv(self, timeout=None):
        if False:
            return 10
        'Receive an ISOTP frame, blocking if none is available in the buffer.'
        return self.rx_queue.recv()