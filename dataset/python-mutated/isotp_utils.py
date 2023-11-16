import struct
from scapy.config import conf
from scapy.utils import EDecimal
from scapy.packet import Packet
from scapy.sessions import DefaultSession
from scapy.supersocket import SuperSocket
from scapy.contrib.isotp.isotp_packet import ISOTP, N_PCI_CF, N_PCI_SF, N_PCI_FF, N_PCI_FC
from typing import cast, Iterable, Iterator, Optional, Union, List, Tuple, Dict, Any, Type

class ISOTPMessageBuilderIter(object):
    """
    Iterator class for ISOTPMessageBuilder
    """
    slots = ['builder']

    def __init__(self, builder):
        if False:
            for i in range(10):
                print('nop')
        self.builder = builder

    def __iter__(self):
        if False:
            return 10
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        while self.builder.count:
            p = self.builder.pop()
            if p is None:
                break
            else:
                return p
        raise StopIteration
    next = __next__

class ISOTPMessageBuilder(object):
    """
    Initialize a ISOTPMessageBuilder object

    Utility class to build ISOTP messages out of CAN frames, used by both
    ISOTP.defragment() and ISOTPSession.

    This class attempts to interpret some CAN frames as ISOTP frames, both with
    and without extended addressing at the same time. For example, if an
    extended address of 07 is being used, all frames will also be interpreted
    as ISOTP single-frame messages.

    CAN frames are fed to an ISOTPMessageBuilder object with the feed() method
    and the resulting ISOTP frames can be extracted using the pop() method.

    :param use_ext_address: True for only attempting to defragment with
                         extended addressing, False for only attempting
                         to defragment without extended addressing,
                         or None for both
    :param rx_id: Destination Identifier
    :param basecls: The class of packets that will be returned,
                    defaults to ISOTP
    """

    class Bucket(object):
        """
        Helper class to store not finished ISOTP messages while building.
        """

        def __init__(self, total_len, first_piece, ts):
            if False:
                for i in range(10):
                    print('nop')
            self.pieces = list()
            self.total_len = total_len
            self.current_len = 0
            self.ready = None
            self.tx_id = None
            self.ext_address = None
            self.time = ts
            self.push(first_piece)

        def push(self, piece):
            if False:
                i = 10
                return i + 15
            self.pieces.append(piece)
            self.current_len += len(piece)
            if self.current_len >= self.total_len:
                isotp_data = b''.join(self.pieces)
                self.ready = isotp_data[:self.total_len]

    def __init__(self, use_ext_address=None, rx_id=None, basecls=ISOTP):
        if False:
            print('Hello World!')
        self.ready = []
        self.buckets = {}
        self.use_ext_addr = use_ext_address
        self.basecls = basecls
        self.rx_ids = None
        self.last_ff = None
        self.last_ff_ex = None
        if rx_id is not None:
            if isinstance(rx_id, list):
                self.rx_ids = rx_id
            elif isinstance(rx_id, int):
                self.rx_ids = [rx_id]
            elif hasattr(rx_id, '__iter__'):
                self.rx_ids = rx_id
            else:
                raise TypeError('Invalid type for argument rx_id!')

    def feed(self, can):
        if False:
            print('Hello World!')
        'Attempt to feed an incoming CAN frame into the state machine'
        if not isinstance(can, Packet) and hasattr(can, '__iter__'):
            for p in can:
                self.feed(p)
            return
        if not isinstance(can, Packet):
            return
        if self.rx_ids is not None and can.identifier not in self.rx_ids:
            return
        data = bytes(can.data)
        if len(data) > 1 and self.use_ext_addr is not True:
            self._try_feed(can.identifier, None, data, can.time)
        if len(data) > 2 and self.use_ext_addr is not False:
            ea = data[0]
            self._try_feed(can.identifier, ea, data[1:], can.time)

    @property
    def count(self):
        if False:
            print('Hello World!')
        'Returns the number of ready ISOTP messages built from the provided\n        can frames\n\n        :return: Number of ready ISOTP messages\n        '
        return len(self.ready)

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.count

    def pop(self, identifier=None, ext_addr=None):
        if False:
            print('Hello World!')
        'Returns a built ISOTP message\n\n        :param identifier: if not None, only return isotp messages with this\n                           destination\n        :param ext_addr: if identifier is not None, only return isotp messages\n                         with this extended address for destination\n        :returns: an ISOTP packet, or None if no message is ready\n        '
        if identifier is not None:
            for i in range(len(self.ready)):
                b = self.ready[i]
                iden = b[0]
                ea = b[1]
                if iden == identifier and ext_addr == ea:
                    return ISOTPMessageBuilder._build(self.ready.pop(i), self.basecls)
            return None
        if len(self.ready) > 0:
            return ISOTPMessageBuilder._build(self.ready.pop(0), self.basecls)
        return None

    def __iter__(self):
        if False:
            while True:
                i = 10
        return ISOTPMessageBuilderIter(self)

    @staticmethod
    def _build(t, basecls=ISOTP):
        if False:
            return 10
        bucket = t[2]
        data = bucket.ready or b''
        try:
            p = basecls(data)
        except Exception:
            if conf.debug_dissector:
                from scapy.sendrecv import debug
                debug.crashed_on = (basecls, data)
            raise
        if hasattr(p, 'rx_id'):
            p.rx_id = t[0]
        if hasattr(p, 'rx_ext_address'):
            p.rx_ext_address = t[1]
        if hasattr(p, 'tx_id'):
            p.tx_id = bucket.tx_id
        if hasattr(p, 'ext_address'):
            p.ext_address = bucket.ext_address
        if hasattr(p, 'time'):
            p.time = bucket.time
        return p

    def _feed_first_frame(self, identifier, ea, data, ts):
        if False:
            return 10
        if len(data) < 3:
            return False
        header = struct.unpack('>H', bytes(data[:2]))[0]
        expected_length = header & 4095
        isotp_data = data[2:]
        if expected_length == 0 and len(data) >= 6:
            expected_length = struct.unpack('>I', bytes(data[2:6]))[0]
            isotp_data = data[6:]
        key = (ea, identifier, 1)
        if ea is None:
            self.last_ff = key
        else:
            self.last_ff_ex = key
        self.buckets[key] = self.Bucket(expected_length, isotp_data, ts)
        return True

    def _feed_single_frame(self, identifier, ea, data, ts):
        if False:
            for i in range(10):
                print('nop')
        if len(data) < 2:
            return False
        length = data[0] & 15
        isotp_data = data[1:length + 1]
        if length > len(isotp_data):
            return False
        self.ready.append((identifier, ea, self.Bucket(length, isotp_data, ts)))
        return True

    def _feed_consecutive_frame(self, identifier, ea, data):
        if False:
            for i in range(10):
                print('nop')
        if len(data) < 2:
            return False
        first_byte = data[0]
        seq_no = first_byte & 15
        isotp_data = data[1:]
        key = (ea, identifier, seq_no)
        bucket = self.buckets.pop(key, None)
        if bucket is None:
            return False
        bucket.push(isotp_data)
        if bucket.ready is None:
            next_seq = (seq_no + 1) % 16
            key = (ea, identifier, next_seq)
            self.buckets[key] = bucket
        else:
            self.ready.append((identifier, ea, bucket))
        return True

    def _feed_flow_control_frame(self, identifier, ea, data):
        if False:
            for i in range(10):
                print('nop')
        if len(data) < 3:
            return False
        keys = [x for x in (self.last_ff, self.last_ff_ex) if x is not None]
        buckets = [self.buckets.pop(k, None) for k in keys]
        self.last_ff = None
        self.last_ff_ex = None
        if not any(buckets) or not any(keys):
            return False
        for (key, bucket) in zip(keys, buckets):
            if bucket is None:
                continue
            bucket.tx_id = identifier
            bucket.ext_address = ea
            self.buckets[key] = bucket
        return True

    def _try_feed(self, identifier, ea, data, ts):
        if False:
            i = 10
            return i + 15
        first_byte = data[0]
        if len(data) > 1 and first_byte & 240 == N_PCI_SF:
            self._feed_single_frame(identifier, ea, data, ts)
        if len(data) > 2 and first_byte & 240 == N_PCI_FF:
            self._feed_first_frame(identifier, ea, data, ts)
        if len(data) > 1 and first_byte & 240 == N_PCI_CF:
            self._feed_consecutive_frame(identifier, ea, data)
        if len(data) > 1 and first_byte & 240 == N_PCI_FC:
            self._feed_flow_control_frame(identifier, ea, data)

class ISOTPSession(DefaultSession):
    """Defragment ISOTP packets 'on-the-flow'.

    Usage:
        >>> sniff(session=ISOTPSession)
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.m = ISOTPMessageBuilder(use_ext_address=kwargs.pop('use_ext_address', None), rx_id=kwargs.pop('rx_id', None), basecls=kwargs.pop('basecls', ISOTP))
        super(ISOTPSession, self).__init__(*args, **kwargs)

    def recv(self, sock: SuperSocket) -> Iterator[Packet]:
        if False:
            print('Hello World!')
        '\n        Will be called by sniff() to ask for a packet\n        '
        pkt = sock.recv()
        if not pkt:
            return
        self.m.feed(pkt)
        while len(self.m) > 0:
            rcvd = cast(Optional[Packet], self.m.pop())
            if rcvd:
                rcvd = self.process(rcvd)
            if rcvd:
                yield rcvd