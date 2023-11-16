import asyncio
import enum
import hmac
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from struct import pack, unpack_from
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Set, Tuple, cast, no_type_check
from google_crc32c import value as crc32c
from pyee.asyncio import AsyncIOEventEmitter
from .exceptions import InvalidStateError
from .rtcdatachannel import RTCDataChannel, RTCDataChannelParameters
from .rtcdtlstransport import RTCDtlsTransport
from .utils import random32, uint16_add, uint16_gt, uint32_gt, uint32_gte
logger = logging.getLogger(__name__)
COOKIE_LENGTH = 24
COOKIE_LIFETIME = 60
MAX_STREAMS = 65535
USERDATA_MAX_LENGTH = 1200
SCTP_CAUSE_INVALID_STREAM = 1
SCTP_CAUSE_STALE_COOKIE = 3
SCTP_DATA_LAST_FRAG = 1
SCTP_DATA_FIRST_FRAG = 2
SCTP_DATA_UNORDERED = 4
SCTP_MAX_ASSOCIATION_RETRANS = 10
SCTP_MAX_BURST = 4
SCTP_MAX_INIT_RETRANS = 8
SCTP_RTO_ALPHA = 1 / 8
SCTP_RTO_BETA = 1 / 4
SCTP_RTO_INITIAL = 3.0
SCTP_RTO_MIN = 1
SCTP_RTO_MAX = 60
SCTP_TSN_MODULO = 2 ** 32
RECONFIG_MAX_STREAMS = 135
SCTP_STATE_COOKIE = 7
SCTP_STR_RESET_OUT_REQUEST = 13
SCTP_STR_RESET_RESPONSE = 16
SCTP_STR_RESET_ADD_OUT_STREAMS = 17
SCTP_SUPPORTED_CHUNK_EXT = 32776
SCTP_PRSCTP_SUPPORTED = 49152
DATA_CHANNEL_ACK = 2
DATA_CHANNEL_OPEN = 3
DATA_CHANNEL_RELIABLE = 0
DATA_CHANNEL_PARTIAL_RELIABLE_REXMIT = 1
DATA_CHANNEL_PARTIAL_RELIABLE_TIMED = 2
DATA_CHANNEL_RELIABLE_UNORDERED = 128
DATA_CHANNEL_PARTIAL_RELIABLE_REXMIT_UNORDERED = 129
DATA_CHANNEL_PARTIAL_RELIABLE_TIMED_UNORDERED = 130
WEBRTC_DCEP = 50
WEBRTC_STRING = 51
WEBRTC_BINARY = 53
WEBRTC_STRING_EMPTY = 56
WEBRTC_BINARY_EMPTY = 57

def chunk_type(chunk) -> str:
    if False:
        for i in range(10):
            print('nop')
    return chunk.__class__.__name__

def decode_params(body: bytes) -> List[Tuple[int, bytes]]:
    if False:
        for i in range(10):
            print('nop')
    params = []
    pos = 0
    while pos <= len(body) - 4:
        (param_type, param_length) = unpack_from('!HH', body, pos)
        params.append((param_type, body[pos + 4:pos + param_length]))
        pos += param_length + padl(param_length)
    return params

def encode_params(params: List[Tuple[int, bytes]]) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    body = b''
    padding = b''
    for (param_type, param_value) in params:
        param_length = len(param_value) + 4
        body += padding
        body += pack('!HH', param_type, param_length) + param_value
        padding = b'\x00' * padl(param_length)
    return body

def padl(length: int) -> int:
    if False:
        print('Hello World!')
    m = length % 4
    return 4 - m if m else 0

def tsn_minus_one(a: int) -> int:
    if False:
        while True:
            i = 10
    return (a - 1) % SCTP_TSN_MODULO

def tsn_plus_one(a: int) -> int:
    if False:
        i = 10
        return i + 15
    return (a + 1) % SCTP_TSN_MODULO

class Chunk:
    type = -1

    def __init__(self, flags: int=0, body: bytes=b'') -> None:
        if False:
            while True:
                i = 10
        self.flags = flags
        self.body = body

    def __bytes__(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        body = self.body
        data = pack('!BBH', self.type, self.flags, len(body) + 4) + body
        data += b'\x00' * padl(len(body))
        return data

    def __repr__(self) -> str:
        if False:
            return 10
        return f'{chunk_type(self)}(flags={self.flags})'

class BaseParamsChunk(Chunk):

    def __init__(self, flags: int=0, body: Optional[bytes]=None) -> None:
        if False:
            return 10
        self.flags = flags
        if body:
            self.params = decode_params(body)
        else:
            self.params = []

    @property
    def body(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        return encode_params(self.params)

class AbortChunk(BaseParamsChunk):
    type = 6

class CookieAckChunk(Chunk):
    type = 11

class CookieEchoChunk(Chunk):
    type = 10

class DataChunk(Chunk):
    type = 0

    def __init__(self, flags: int=0, body: Optional[bytes]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.flags = flags
        if body:
            (self.tsn, self.stream_id, self.stream_seq, self.protocol) = unpack_from('!LHHL', body)
            self.user_data = body[12:]
        else:
            self.tsn = 0
            self.stream_id = 0
            self.stream_seq = 0
            self.protocol = 0
            self.user_data = b''

    def __bytes__(self) -> bytes:
        if False:
            while True:
                i = 10
        length = 16 + len(self.user_data)
        data = pack('!BBHLHHL', self.type, self.flags, length, self.tsn, self.stream_id, self.stream_seq, self.protocol) + self.user_data
        if length % 4:
            data += b'\x00' * padl(length)
        return data

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'DataChunk(flags={self.flags}, tsn={self.tsn}, stream_id={self.stream_id}, stream_seq={self.stream_seq})'

class ErrorChunk(BaseParamsChunk):
    type = 9

class ForwardTsnChunk(Chunk):
    type = 192

    def __init__(self, flags: int=0, body: Optional[bytes]=None) -> None:
        if False:
            print('Hello World!')
        self.flags = flags
        self.streams: List[Tuple[int, int]] = []
        if body:
            self.cumulative_tsn = unpack_from('!L', body, 0)[0]
            pos = 4
            while pos < len(body):
                self.streams.append(cast(Tuple[int, int], unpack_from('!HH', body, pos)))
                pos += 4
        else:
            self.cumulative_tsn = 0

    @property
    def body(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        body = pack('!L', self.cumulative_tsn)
        for (stream_id, stream_seq) in self.streams:
            body += pack('!HH', stream_id, stream_seq)
        return body

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'ForwardTsnChunk(cumulative_tsn={self.cumulative_tsn}, streams={self.streams})'

class HeartbeatChunk(BaseParamsChunk):
    type = 4

class HeartbeatAckChunk(BaseParamsChunk):
    type = 5

class BaseInitChunk(Chunk):

    def __init__(self, flags: int=0, body: Optional[bytes]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.flags = flags
        if body:
            (self.initiate_tag, self.advertised_rwnd, self.outbound_streams, self.inbound_streams, self.initial_tsn) = unpack_from('!LLHHL', body)
            self.params = decode_params(body[16:])
        else:
            self.initiate_tag = 0
            self.advertised_rwnd = 0
            self.outbound_streams = 0
            self.inbound_streams = 0
            self.initial_tsn = 0
            self.params = []

    @property
    def body(self) -> bytes:
        if False:
            while True:
                i = 10
        body = pack('!LLHHL', self.initiate_tag, self.advertised_rwnd, self.outbound_streams, self.inbound_streams, self.initial_tsn)
        body += encode_params(self.params)
        return body

class InitChunk(BaseInitChunk):
    type = 1

class InitAckChunk(BaseInitChunk):
    type = 2

class ReconfigChunk(BaseParamsChunk):
    type = 130

class SackChunk(Chunk):
    type = 3

    def __init__(self, flags=0, body=None):
        if False:
            print('Hello World!')
        self.flags = flags
        self.gaps = []
        self.duplicates = []
        if body:
            (self.cumulative_tsn, self.advertised_rwnd, nb_gaps, nb_duplicates) = unpack_from('!LLHH', body)
            pos = 12
            for i in range(nb_gaps):
                self.gaps.append(unpack_from('!HH', body, pos))
                pos += 4
            for i in range(nb_duplicates):
                self.duplicates.append(unpack_from('!L', body, pos)[0])
                pos += 4
        else:
            self.cumulative_tsn = 0
            self.advertised_rwnd = 0

    def __bytes__(self) -> bytes:
        if False:
            i = 10
            return i + 15
        length = 16 + 4 * (len(self.gaps) + len(self.duplicates))
        data = pack('!BBHLLHH', self.type, self.flags, length, self.cumulative_tsn, self.advertised_rwnd, len(self.gaps), len(self.duplicates))
        for gap in self.gaps:
            data += pack('!HH', *gap)
        for tsn in self.duplicates:
            data += pack('!L', tsn)
        return data

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'SackChunk(flags={self.flags}, advertised_rwnd={self.advertised_rwnd}, cumulative_tsn={self.cumulative_tsn}, gaps={self.gaps})'

class ShutdownChunk(Chunk):
    type = 7

    def __init__(self, flags=0, body=None):
        if False:
            for i in range(10):
                print('nop')
        self.flags = flags
        if body:
            self.cumulative_tsn = unpack_from('!L', body)[0]
        else:
            self.cumulative_tsn = 0

    @property
    def body(self) -> bytes:
        if False:
            while True:
                i = 10
        return pack('!L', self.cumulative_tsn)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'ShutdownChunk(flags={self.flags}, cumulative_tsn={self.cumulative_tsn})'

class ShutdownAckChunk(Chunk):
    type = 8

class ShutdownCompleteChunk(Chunk):
    type = 14
CHUNK_CLASSES = [DataChunk, InitChunk, InitAckChunk, SackChunk, HeartbeatChunk, HeartbeatAckChunk, AbortChunk, ShutdownChunk, ShutdownAckChunk, ErrorChunk, CookieEchoChunk, CookieAckChunk, ShutdownCompleteChunk, ReconfigChunk, ForwardTsnChunk]
CHUNK_TYPES = dict(((cls.type, cls) for cls in CHUNK_CLASSES))

def parse_packet(data: bytes) -> Tuple[int, int, int, List[Any]]:
    if False:
        return 10
    length = len(data)
    if length < 12:
        raise ValueError('SCTP packet length is less than 12 bytes')
    (source_port, destination_port, verification_tag) = unpack_from('!HHL', data)
    checksum = unpack_from('<L', data, 8)[0]
    if checksum != crc32c(data[0:8] + b'\x00\x00\x00\x00' + data[12:]):
        raise ValueError('SCTP packet has invalid checksum')
    chunks = []
    pos = 12
    while pos <= length - 4:
        (chunk_type, chunk_flags, chunk_length) = unpack_from('!BBH', data, pos)
        chunk_body = data[pos + 4:pos + chunk_length]
        chunk_cls = CHUNK_TYPES.get(chunk_type)
        if chunk_cls:
            chunks.append(chunk_cls(flags=chunk_flags, body=chunk_body))
        pos += chunk_length + padl(chunk_length)
    return (source_port, destination_port, verification_tag, chunks)

def serialize_packet(source_port: int, destination_port: int, verification_tag: int, chunk: Chunk) -> bytes:
    if False:
        print('Hello World!')
    header = pack('!HHL', source_port, destination_port, verification_tag)
    data = bytes(chunk)
    checksum = crc32c(header + b'\x00\x00\x00\x00' + data)
    return header + pack('<L', checksum) + data

@dataclass
class StreamResetOutgoingParam:
    request_sequence: int
    response_sequence: int
    last_tsn: int
    streams: List[bytes] = field(default_factory=list)

    def __bytes__(self) -> bytes:
        if False:
            return 10
        data = pack('!LLL', self.request_sequence, self.response_sequence, self.last_tsn)
        for stream in self.streams:
            data += pack('!H', stream)
        return data

    @classmethod
    def parse(cls, data):
        if False:
            return 10
        (request_sequence, response_sequence, last_tsn) = unpack_from('!LLL', data)
        streams = []
        for pos in range(12, len(data), 2):
            streams.append(unpack_from('!H', data, pos)[0])
        return cls(request_sequence=request_sequence, response_sequence=response_sequence, last_tsn=last_tsn, streams=streams)

@dataclass
class StreamAddOutgoingParam:
    request_sequence: int
    new_streams: int

    def __bytes__(self) -> bytes:
        if False:
            while True:
                i = 10
        data = pack('!LHH', self.request_sequence, self.new_streams, 0)
        return data

    @classmethod
    def parse(cls, data):
        if False:
            for i in range(10):
                print('nop')
        (request_sequence, new_streams, reserved) = unpack_from('!LHH', data)
        return cls(request_sequence=request_sequence, new_streams=new_streams)

@dataclass
class StreamResetResponseParam:
    response_sequence: int
    result: int

    def __bytes__(self) -> bytes:
        if False:
            print('Hello World!')
        return pack('!LL', self.response_sequence, self.result)

    @classmethod
    def parse(cls, data):
        if False:
            for i in range(10):
                print('nop')
        (response_sequence, result) = unpack_from('!LL', data)
        return cls(response_sequence=response_sequence, result=result)
RECONFIG_PARAM_TYPES = {13: StreamResetOutgoingParam, 16: StreamResetResponseParam, 17: StreamAddOutgoingParam}

class InboundStream:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.reassembly: List[DataChunk] = []
        self.sequence_number = 0

    def add_chunk(self, chunk: DataChunk) -> None:
        if False:
            return 10
        if not self.reassembly or uint32_gt(chunk.tsn, self.reassembly[-1].tsn):
            self.reassembly.append(chunk)
            return
        for (i, rchunk) in enumerate(self.reassembly):
            assert rchunk.tsn != chunk.tsn, 'duplicate chunk in reassembly'
            if uint32_gt(rchunk.tsn, chunk.tsn):
                self.reassembly.insert(i, chunk)
                break

    def pop_messages(self) -> Iterator[Tuple[int, int, bytes]]:
        if False:
            print('Hello World!')
        pos = 0
        start_pos = None
        while pos < len(self.reassembly):
            chunk = self.reassembly[pos]
            if start_pos is None:
                ordered = not chunk.flags & SCTP_DATA_UNORDERED
                if not chunk.flags & SCTP_DATA_FIRST_FRAG:
                    if ordered:
                        break
                    else:
                        pos += 1
                        continue
                if ordered and uint16_gt(chunk.stream_seq, self.sequence_number):
                    break
                expected_tsn = chunk.tsn
                start_pos = pos
            elif chunk.tsn != expected_tsn:
                if ordered:
                    break
                else:
                    start_pos = None
                    pos += 1
                    continue
            if chunk.flags & SCTP_DATA_LAST_FRAG:
                user_data = b''.join([c.user_data for c in self.reassembly[start_pos:pos + 1]])
                self.reassembly = self.reassembly[:start_pos] + self.reassembly[pos + 1:]
                if ordered and chunk.stream_seq == self.sequence_number:
                    self.sequence_number = uint16_add(self.sequence_number, 1)
                pos = start_pos
                yield (chunk.stream_id, chunk.protocol, user_data)
            else:
                pos += 1
            expected_tsn = tsn_plus_one(expected_tsn)

    def prune_chunks(self, tsn: int) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Prune chunks up to the given TSN.\n        '
        pos = -1
        size = 0
        for (i, chunk) in enumerate(self.reassembly):
            if uint32_gte(tsn, chunk.tsn):
                pos = i
                size += len(chunk.user_data)
            else:
                break
        self.reassembly = self.reassembly[pos + 1:]
        return size

@dataclass
class RTCSctpCapabilities:
    """
    The :class:`RTCSctpCapabilities` dictionary provides information about the
    capabilities of the :class:`RTCSctpTransport`.
    """
    maxMessageSize: int
    '\n    The maximum size of data that the implementation can send or\n    0 if the implementation can handle messages of any size.\n\n    '

class RTCSctpTransport(AsyncIOEventEmitter):
    """
    The :class:`RTCSctpTransport` interface includes information relating to
    Stream Control Transmission Protocol (SCTP) transport.

    :param transport: An :class:`RTCDtlsTransport`.
    """

    def __init__(self, transport: RTCDtlsTransport, port: int=5000) -> None:
        if False:
            print('Hello World!')
        if transport.state == 'closed':
            raise InvalidStateError
        super().__init__()
        self._association_state = self.State.CLOSED
        self.__log_debug: Callable[..., None] = lambda *args: None
        self.__started = False
        self.__state = 'new'
        self.__transport = transport
        self._loop = asyncio.get_event_loop()
        self._hmac_key = os.urandom(16)
        self._local_partial_reliability = True
        self._local_port = port
        self._local_verification_tag = random32()
        self._remote_extensions: List[int] = []
        self._remote_partial_reliability = False
        self._remote_port: Optional[int] = None
        self._remote_verification_tag = 0
        self._advertised_rwnd = 1024 * 1024
        self._inbound_streams: Dict[int, InboundStream] = {}
        self._inbound_streams_count = 0
        self._inbound_streams_max = MAX_STREAMS
        self._last_received_tsn: Optional[int] = None
        self._sack_duplicates: List[int] = []
        self._sack_misordered: Set[int] = set()
        self._sack_needed = False
        self._cwnd = 3 * USERDATA_MAX_LENGTH
        self._fast_recovery_exit = None
        self._fast_recovery_transmit = False
        self._forward_tsn_chunk: Optional[ForwardTsnChunk] = None
        self._flight_size = 0
        self._local_tsn = random32()
        self._last_sacked_tsn = tsn_minus_one(self._local_tsn)
        self._advanced_peer_ack_tsn = tsn_minus_one(self._local_tsn)
        self._outbound_queue: Deque[DataChunk] = deque()
        self._outbound_stream_seq: Dict[int, int] = {}
        self._outbound_streams_count = MAX_STREAMS
        self._partial_bytes_acked = 0
        self._sent_queue: Deque[DataChunk] = deque()
        self._reconfig_queue: List[int] = []
        self._reconfig_request = None
        self._reconfig_request_seq = self._local_tsn
        self._reconfig_response_seq = 0
        self._srtt: Optional[float] = None
        self._rttvar: Optional[float] = None
        self._rto = SCTP_RTO_INITIAL
        self._t1_chunk: Optional[Chunk] = None
        self._t1_failures = 0
        self._t1_handle: Optional[asyncio.TimerHandle] = None
        self._t2_chunk: Optional[Chunk] = None
        self._t2_failures = 0
        self._t2_handle: Optional[asyncio.TimerHandle] = None
        self._t3_handle: Optional[asyncio.TimerHandle] = None
        self._data_channel_id: Optional[int] = None
        self._data_channel_queue: Deque[Tuple[RTCDataChannel, int, bytes]] = deque()
        self._data_channels: Dict[int, RTCDataChannel] = {}
        self._bundled = False
        self.mid: Optional[str] = None

    @property
    def is_server(self) -> bool:
        if False:
            return 10
        return self.transport.transport.role != 'controlling'

    @property
    def maxChannels(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        '\n        The maximum number of :class:`RTCDataChannel` that can be used simultaneously.\n        '
        if self._inbound_streams_count:
            return min(self._inbound_streams_count, self._outbound_streams_count)
        return None

    @property
    def port(self) -> int:
        if False:
            return 10
        '\n        The local SCTP port number used for data channels.\n        '
        return self._local_port

    @property
    def state(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        The current state of the SCTP transport.\n        '
        return self.__state

    @property
    def transport(self):
        if False:
            print('Hello World!')
        '\n        The :class:`RTCDtlsTransport` over which SCTP data is transmitted.\n        '
        return self.__transport

    @classmethod
    def getCapabilities(cls) -> RTCSctpCapabilities:
        if False:
            i = 10
            return i + 15
        '\n        Retrieve the capabilities of the transport.\n\n        :rtype: RTCSctpCapabilities\n        '
        return RTCSctpCapabilities(maxMessageSize=65536)

    def setTransport(self, transport) -> None:
        if False:
            print('Hello World!')
        self.__transport = transport

    async def start(self, remoteCaps: RTCSctpCapabilities, remotePort: int) -> None:
        """
        Start the transport.
        """
        if not self.__started:
            self.__started = True
            self.__state = 'connecting'
            self._remote_port = remotePort
            if logger.isEnabledFor(logging.DEBUG):
                prefix = 'RTCSctpTransport(%s) ' % (self.is_server and 'server' or 'client')
                self.__log_debug = lambda msg, *args: logger.debug(prefix + msg, *args)
            if self.is_server:
                self._data_channel_id = 0
            else:
                self._data_channel_id = 1
            self.__transport._register_data_receiver(self)
            if not self.is_server:
                await self._init()

    async def stop(self) -> None:
        """
        Stop the transport.
        """
        if self._association_state != self.State.CLOSED:
            await self._abort()
        self.__transport._unregister_data_receiver(self)
        self._set_state(self.State.CLOSED)

    async def _abort(self) -> None:
        """
        Abort the association.
        """
        chunk = AbortChunk()
        try:
            await self._send_chunk(chunk)
        except ConnectionError:
            pass

    async def _init(self) -> None:
        """
        Initialize the association.
        """
        chunk = InitChunk()
        chunk.initiate_tag = self._local_verification_tag
        chunk.advertised_rwnd = self._advertised_rwnd
        chunk.outbound_streams = self._outbound_streams_count
        chunk.inbound_streams = self._inbound_streams_max
        chunk.initial_tsn = self._local_tsn
        self._set_extensions(chunk.params)
        await self._send_chunk(chunk)
        self._t1_start(chunk)
        self._set_state(self.State.COOKIE_WAIT)

    def _flight_size_decrease(self, chunk: DataChunk) -> None:
        if False:
            print('Hello World!')
        self._flight_size = max(0, self._flight_size - chunk._book_size)

    def _flight_size_increase(self, chunk: DataChunk) -> None:
        if False:
            print('Hello World!')
        self._flight_size += chunk._book_size

    def _get_extensions(self, params: List[Tuple[int, bytes]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets what extensions are supported by the remote party.\n        '
        for (k, v) in params:
            if k == SCTP_PRSCTP_SUPPORTED:
                self._remote_partial_reliability = True
            elif k == SCTP_SUPPORTED_CHUNK_EXT:
                self._remote_extensions = list(v)

    def _set_extensions(self, params: List[Tuple[int, bytes]]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Sets what extensions are supported by the local party.\n        '
        extensions = []
        if self._local_partial_reliability:
            params.append((SCTP_PRSCTP_SUPPORTED, b''))
            extensions.append(ForwardTsnChunk.type)
        extensions.append(ReconfigChunk.type)
        params.append((SCTP_SUPPORTED_CHUNK_EXT, bytes(extensions)))

    def _get_inbound_stream(self, stream_id: int) -> InboundStream:
        if False:
            return 10
        '\n        Get or create the inbound stream with the specified ID.\n        '
        if stream_id not in self._inbound_streams:
            self._inbound_streams[stream_id] = InboundStream()
        return self._inbound_streams[stream_id]

    def _get_timestamp(self) -> int:
        if False:
            return 10
        return int(time.time())

    async def _handle_data(self, data):
        """
        Handle data received from the network.
        """
        try:
            (_, _, verification_tag, chunks) = parse_packet(data)
        except ValueError:
            return
        init_chunk = len([x for x in chunks if isinstance(x, InitChunk)])
        if init_chunk:
            assert len(chunks) == 1
            expected_tag = 0
        else:
            expected_tag = self._local_verification_tag
        if verification_tag != expected_tag:
            self.__log_debug('Bad verification tag %d vs %d', verification_tag, expected_tag)
            return
        for chunk in chunks:
            await self._receive_chunk(chunk)
        if self._sack_needed:
            await self._send_sack()

    @no_type_check
    def _maybe_abandon(self, chunk: DataChunk) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine if a chunk needs to be marked as abandoned.\n\n        If it does, it marks the chunk and any other chunk belong to the same\n        message as abandoned.\n        '
        if chunk._abandoned:
            return True
        abandon = chunk._max_retransmits is not None and chunk._sent_count > chunk._max_retransmits or (chunk._expiry is not None and chunk._expiry < time.time())
        if not abandon:
            return False
        chunk_pos = self._sent_queue.index(chunk)
        for pos in range(chunk_pos, -1, -1):
            ochunk = self._sent_queue[pos]
            ochunk._abandoned = True
            ochunk._retransmit = False
            if ochunk.flags & SCTP_DATA_FIRST_FRAG:
                break
        for pos in range(chunk_pos, len(self._sent_queue)):
            ochunk = self._sent_queue[pos]
            ochunk._abandoned = True
            ochunk._retransmit = False
            if ochunk.flags & SCTP_DATA_LAST_FRAG:
                break
        return True

    def _mark_received(self, tsn: int) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Mark an incoming data TSN as received.\n        '
        if uint32_gte(self._last_received_tsn, tsn) or tsn in self._sack_misordered:
            self._sack_duplicates.append(tsn)
            return True
        self._sack_misordered.add(tsn)
        for tsn in sorted(self._sack_misordered):
            if tsn == tsn_plus_one(self._last_received_tsn):
                self._last_received_tsn = tsn
            else:
                break

        def is_obsolete(x):
            if False:
                for i in range(10):
                    print('nop')
            return uint32_gt(x, self._last_received_tsn)
        self._sack_duplicates = list(filter(is_obsolete, self._sack_duplicates))
        self._sack_misordered = set(filter(is_obsolete, self._sack_misordered))
        return False

    async def _receive(self, stream_id: int, pp_id: int, data: bytes) -> None:
        """
        Receive data stream -> ULP.
        """
        await self._data_channel_receive(stream_id, pp_id, data)

    async def _receive_chunk(self, chunk):
        """
        Handle an incoming chunk.
        """
        self.__log_debug('< %s', chunk)
        if isinstance(chunk, DataChunk):
            await self._receive_data_chunk(chunk)
        elif isinstance(chunk, SackChunk):
            await self._receive_sack_chunk(chunk)
        elif isinstance(chunk, ForwardTsnChunk):
            await self._receive_forward_tsn_chunk(chunk)
        elif isinstance(chunk, HeartbeatChunk):
            ack = HeartbeatAckChunk()
            ack.params = chunk.params
            await self._send_chunk(ack)
        elif isinstance(chunk, AbortChunk):
            self.__log_debug('x Association was aborted by remote party')
            self._set_state(self.State.CLOSED)
        elif isinstance(chunk, ShutdownChunk):
            self._t2_cancel()
            self._set_state(self.State.SHUTDOWN_RECEIVED)
            ack = ShutdownAckChunk()
            await self._send_chunk(ack)
            self._t2_start(ack)
            self._set_state(self.State.SHUTDOWN_ACK_SENT)
        elif isinstance(chunk, ShutdownCompleteChunk) and self._association_state == self.State.SHUTDOWN_ACK_SENT:
            self._t2_cancel()
            self._set_state(self.State.CLOSED)
        elif isinstance(chunk, ReconfigChunk) and self._association_state == self.State.ESTABLISHED:
            for param in chunk.params:
                cls = RECONFIG_PARAM_TYPES.get(param[0])
                if cls:
                    await self._receive_reconfig_param(cls.parse(param[1]))
        elif isinstance(chunk, InitChunk) and self.is_server:
            self._last_received_tsn = tsn_minus_one(chunk.initial_tsn)
            self._reconfig_response_seq = tsn_minus_one(chunk.initial_tsn)
            self._remote_verification_tag = chunk.initiate_tag
            self._ssthresh = chunk.advertised_rwnd
            self._get_extensions(chunk.params)
            self.__log_debug('- Peer supports %d outbound streams, %d max inbound streams', chunk.outbound_streams, chunk.inbound_streams)
            self._inbound_streams_count = min(chunk.outbound_streams, self._inbound_streams_max)
            self._outbound_streams_count = min(self._outbound_streams_count, chunk.inbound_streams)
            ack = InitAckChunk()
            ack.initiate_tag = self._local_verification_tag
            ack.advertised_rwnd = self._advertised_rwnd
            ack.outbound_streams = self._outbound_streams_count
            ack.inbound_streams = self._inbound_streams_max
            ack.initial_tsn = self._local_tsn
            self._set_extensions(ack.params)
            cookie = pack('!L', self._get_timestamp())
            cookie += hmac.new(self._hmac_key, cookie, 'sha1').digest()
            ack.params.append((SCTP_STATE_COOKIE, cookie))
            await self._send_chunk(ack)
        elif isinstance(chunk, CookieEchoChunk) and self.is_server:
            cookie = chunk.body
            if len(cookie) != COOKIE_LENGTH or hmac.new(self._hmac_key, cookie[0:4], 'sha1').digest() != cookie[4:]:
                self.__log_debug('x State cookie is invalid')
                return
            now = self._get_timestamp()
            stamp = unpack_from('!L', cookie)[0]
            if stamp < now - COOKIE_LIFETIME or stamp > now:
                self.__log_debug('x State cookie has expired')
                error = ErrorChunk()
                error.params.append((SCTP_CAUSE_STALE_COOKIE, b'\x00' * 8))
                await self._send_chunk(error)
                return
            ack = CookieAckChunk()
            await self._send_chunk(ack)
            self._set_state(self.State.ESTABLISHED)
        elif isinstance(chunk, InitAckChunk) and self._association_state == self.State.COOKIE_WAIT:
            self._t1_cancel()
            self._last_received_tsn = tsn_minus_one(chunk.initial_tsn)
            self._reconfig_response_seq = tsn_minus_one(chunk.initial_tsn)
            self._remote_verification_tag = chunk.initiate_tag
            self._ssthresh = chunk.advertised_rwnd
            self._get_extensions(chunk.params)
            self.__log_debug('- Peer supports %d outbound streams, %d max inbound streams', chunk.outbound_streams, chunk.inbound_streams)
            self._inbound_streams_count = min(chunk.outbound_streams, self._inbound_streams_max)
            self._outbound_streams_count = min(self._outbound_streams_count, chunk.inbound_streams)
            echo = CookieEchoChunk()
            for (k, v) in chunk.params:
                if k == SCTP_STATE_COOKIE:
                    echo.body = v
                    break
            await self._send_chunk(echo)
            self._t1_start(echo)
            self._set_state(self.State.COOKIE_ECHOED)
        elif isinstance(chunk, CookieAckChunk) and self._association_state == self.State.COOKIE_ECHOED:
            self._t1_cancel()
            self._set_state(self.State.ESTABLISHED)
        elif isinstance(chunk, ErrorChunk) and self._association_state in [self.State.COOKIE_WAIT, self.State.COOKIE_ECHOED]:
            self._t1_cancel()
            self._set_state(self.State.CLOSED)
            self.__log_debug('x Could not establish association')
            return

    async def _receive_data_chunk(self, chunk: DataChunk) -> None:
        """
        Handle a DATA chunk.
        """
        self._sack_needed = True
        if self._mark_received(chunk.tsn):
            return
        inbound_stream = self._get_inbound_stream(chunk.stream_id)
        inbound_stream.add_chunk(chunk)
        self._advertised_rwnd -= len(chunk.user_data)
        for message in inbound_stream.pop_messages():
            self._advertised_rwnd += len(message[2])
            await self._receive(*message)

    async def _receive_forward_tsn_chunk(self, chunk: ForwardTsnChunk) -> None:
        """
        Handle a FORWARD TSN chunk.
        """
        self._sack_needed = True
        if uint32_gte(self._last_received_tsn, chunk.cumulative_tsn):
            return

        def is_obsolete(x):
            if False:
                while True:
                    i = 10
            return uint32_gt(x, self._last_received_tsn)
        self._last_received_tsn = chunk.cumulative_tsn
        self._sack_misordered = set(filter(is_obsolete, self._sack_misordered))
        for tsn in sorted(self._sack_misordered):
            if tsn == tsn_plus_one(self._last_received_tsn):
                self._last_received_tsn = tsn
            else:
                break
        self._sack_duplicates = list(filter(is_obsolete, self._sack_duplicates))
        self._sack_misordered = set(filter(is_obsolete, self._sack_misordered))
        for (stream_id, stream_seq) in chunk.streams:
            inbound_stream = self._get_inbound_stream(stream_id)
            inbound_stream.sequence_number = uint16_add(stream_seq, 1)
            for message in inbound_stream.pop_messages():
                self._advertised_rwnd += len(message[2])
                await self._receive(*message)
        for (stream_id, inbound_stream) in self._inbound_streams.items():
            self._advertised_rwnd += inbound_stream.prune_chunks(self._last_received_tsn)

    @no_type_check
    async def _receive_sack_chunk(self, chunk: SackChunk) -> None:
        """
        Handle a SACK chunk.
        """
        if uint32_gt(self._last_sacked_tsn, chunk.cumulative_tsn):
            return
        received_time = time.time()
        self._last_sacked_tsn = chunk.cumulative_tsn
        cwnd_fully_utilized = self._flight_size >= self._cwnd
        done = 0
        done_bytes = 0
        while self._sent_queue and uint32_gte(self._last_sacked_tsn, self._sent_queue[0].tsn):
            schunk = self._sent_queue.popleft()
            done += 1
            if not schunk._acked:
                done_bytes += schunk._book_size
                self._flight_size_decrease(schunk)
            if done == 1 and schunk._sent_count == 1:
                self._update_rto(received_time - schunk._sent_time)
        loss = False
        if chunk.gaps:
            seen = set()
            for gap in chunk.gaps:
                for pos in range(gap[0], gap[1] + 1):
                    highest_seen_tsn = (chunk.cumulative_tsn + pos) % SCTP_TSN_MODULO
                    seen.add(highest_seen_tsn)
            highest_newly_acked = chunk.cumulative_tsn
            for schunk in self._sent_queue:
                if uint32_gt(schunk.tsn, highest_seen_tsn):
                    break
                if schunk.tsn in seen and (not schunk._acked):
                    done_bytes += schunk._book_size
                    schunk._acked = True
                    self._flight_size_decrease(schunk)
                    highest_newly_acked = schunk.tsn
            for schunk in self._sent_queue:
                if uint32_gt(schunk.tsn, highest_newly_acked):
                    break
                if schunk.tsn not in seen:
                    schunk._misses += 1
                    if schunk._misses == 3:
                        schunk._misses = 0
                        if not self._maybe_abandon(schunk):
                            schunk._retransmit = True
                        schunk._acked = False
                        self._flight_size_decrease(schunk)
                        loss = True
        if self._fast_recovery_exit is None:
            if done and cwnd_fully_utilized:
                if self._cwnd <= self._ssthresh:
                    self._cwnd += min(done_bytes, USERDATA_MAX_LENGTH)
                else:
                    self._partial_bytes_acked += done_bytes
                    if self._partial_bytes_acked >= self._cwnd:
                        self._partial_bytes_acked -= self._cwnd
                        self._cwnd += USERDATA_MAX_LENGTH
            if loss:
                self._ssthresh = max(self._cwnd // 2, 4 * USERDATA_MAX_LENGTH)
                self._cwnd = self._ssthresh
                self._partial_bytes_acked = 0
                self._fast_recovery_exit = self._sent_queue[-1].tsn
                self._fast_recovery_transmit = True
        elif uint32_gte(chunk.cumulative_tsn, self._fast_recovery_exit):
            self._fast_recovery_exit = None
        if not self._sent_queue:
            self._t3_cancel()
        elif done:
            self._t3_restart()
        self._update_advanced_peer_ack_point()
        await self._data_channel_flush()
        await self._transmit()

    async def _receive_reconfig_param(self, param):
        """
        Handle a RE-CONFIG parameter.
        """
        self.__log_debug('<< %s', param)
        if isinstance(param, StreamResetOutgoingParam):
            for stream_id in param.streams:
                self._inbound_streams.pop(stream_id, None)
                channel = self._data_channels.get(stream_id)
                if channel:
                    self._data_channel_close(channel)
            response_param = StreamResetResponseParam(response_sequence=param.request_sequence, result=1)
            self._reconfig_response_seq = param.request_sequence
            await self._send_reconfig_param(response_param)
        elif isinstance(param, StreamAddOutgoingParam):
            self._inbound_streams_count += param.new_streams
            response_param = StreamResetResponseParam(response_sequence=param.request_sequence, result=1)
            self._reconfig_response_seq = param.request_sequence
            await self._send_reconfig_param(response_param)
        elif isinstance(param, StreamResetResponseParam):
            if self._reconfig_request and param.response_sequence == self._reconfig_request.request_sequence:
                for stream_id in self._reconfig_request.streams:
                    self._outbound_stream_seq.pop(stream_id, None)
                    self._data_channel_closed(stream_id)
                self._reconfig_request = None
                await self._transmit_reconfig()

    @no_type_check
    async def _send(self, stream_id: int, pp_id: int, user_data: bytes, expiry: Optional[float]=None, max_retransmits: Optional[int]=None, ordered: bool=True) -> None:
        """
        Send data ULP -> stream.
        """
        if ordered:
            stream_seq = self._outbound_stream_seq.get(stream_id, 0)
        else:
            stream_seq = 0
        fragments = math.ceil(len(user_data) / USERDATA_MAX_LENGTH)
        pos = 0
        for fragment in range(0, fragments):
            chunk = DataChunk()
            chunk.flags = 0
            if not ordered:
                chunk.flags = SCTP_DATA_UNORDERED
            if fragment == 0:
                chunk.flags |= SCTP_DATA_FIRST_FRAG
            if fragment == fragments - 1:
                chunk.flags |= SCTP_DATA_LAST_FRAG
            chunk.tsn = self._local_tsn
            chunk.stream_id = stream_id
            chunk.stream_seq = stream_seq
            chunk.protocol = pp_id
            chunk.user_data = user_data[pos:pos + USERDATA_MAX_LENGTH]
            chunk._abandoned = False
            chunk._acked = False
            chunk._book_size = len(chunk.user_data)
            chunk._expiry = expiry
            chunk._max_retransmits = max_retransmits
            chunk._misses = 0
            chunk._retransmit = False
            chunk._sent_count = 0
            chunk._sent_time = None
            pos += USERDATA_MAX_LENGTH
            self._local_tsn = tsn_plus_one(self._local_tsn)
            self._outbound_queue.append(chunk)
        if ordered:
            self._outbound_stream_seq[stream_id] = uint16_add(stream_seq, 1)
        await self._transmit()

    async def _send_chunk(self, chunk: Chunk) -> None:
        """
        Transmit a chunk (no bundling for now).
        """
        self.__log_debug('> %s', chunk)
        await self.__transport._send_data(serialize_packet(self._local_port, self._remote_port, self._remote_verification_tag, chunk))

    async def _send_reconfig_param(self, param):
        chunk = ReconfigChunk()
        for (k, cls) in RECONFIG_PARAM_TYPES.items():
            if isinstance(param, cls):
                param_type = k
                break
        chunk.params.append((param_type, bytes(param)))
        self.__log_debug('>> %s', param)
        await self._send_chunk(chunk)

    async def _send_sack(self):
        """
        Build and send a selective acknowledgement (SACK) chunk.
        """
        gaps = []
        gap_next = None
        for tsn in sorted(self._sack_misordered):
            pos = (tsn - self._last_received_tsn) % SCTP_TSN_MODULO
            if tsn == gap_next:
                gaps[-1][1] = pos
            else:
                gaps.append([pos, pos])
            gap_next = tsn_plus_one(tsn)
        sack = SackChunk()
        sack.cumulative_tsn = self._last_received_tsn
        sack.advertised_rwnd = max(0, self._advertised_rwnd)
        sack.duplicates = self._sack_duplicates[:]
        sack.gaps = [tuple(x) for x in gaps]
        await self._send_chunk(sack)
        self._sack_duplicates.clear()
        self._sack_needed = False

    def _set_state(self, state) -> None:
        if False:
            while True:
                i = 10
        '\n        Transition the SCTP association to a new state.\n        '
        if state != self._association_state:
            self.__log_debug('- %s -> %s', self._association_state, state)
            self._association_state = state
        if state == self.State.ESTABLISHED:
            self.__state = 'connected'
            for channel in list(self._data_channels.values()):
                if channel.negotiated and channel.readyState != 'open':
                    channel._setReadyState('open')
            asyncio.ensure_future(self._data_channel_flush())
        elif state == self.State.CLOSED:
            self._t1_cancel()
            self._t2_cancel()
            self._t3_cancel()
            self.__state = 'closed'
            for stream_id in list(self._data_channels.keys()):
                self._data_channel_closed(stream_id)
            self.remove_all_listeners()

    def _t1_cancel(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._t1_handle is not None:
            self.__log_debug('- T1(%s) cancel', chunk_type(self._t1_chunk))
            self._t1_handle.cancel()
            self._t1_handle = None
            self._t1_chunk = None

    def _t1_expired(self) -> None:
        if False:
            while True:
                i = 10
        self._t1_failures += 1
        self._t1_handle = None
        self.__log_debug('x T1(%s) expired %d', chunk_type(self._t1_chunk), self._t1_failures)
        if self._t1_failures > SCTP_MAX_INIT_RETRANS:
            self._set_state(self.State.CLOSED)
        else:
            asyncio.ensure_future(self._send_chunk(self._t1_chunk))
            self._t1_handle = self._loop.call_later(self._rto, self._t1_expired)

    def _t1_start(self, chunk: Chunk) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert self._t1_handle is None
        self._t1_chunk = chunk
        self._t1_failures = 0
        self.__log_debug('- T1(%s) start', chunk_type(self._t1_chunk))
        self._t1_handle = self._loop.call_later(self._rto, self._t1_expired)

    def _t2_cancel(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._t2_handle is not None:
            self.__log_debug('- T2(%s) cancel', chunk_type(self._t2_chunk))
            self._t2_handle.cancel()
            self._t2_handle = None
            self._t2_chunk = None

    def _t2_expired(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._t2_failures += 1
        self._t2_handle = None
        self.__log_debug('x T2(%s) expired %d', chunk_type(self._t2_chunk), self._t2_failures)
        if self._t2_failures > SCTP_MAX_ASSOCIATION_RETRANS:
            self._set_state(self.State.CLOSED)
        else:
            asyncio.ensure_future(self._send_chunk(self._t2_chunk))
            self._t2_handle = self._loop.call_later(self._rto, self._t2_expired)

    def _t2_start(self, chunk) -> None:
        if False:
            i = 10
            return i + 15
        assert self._t2_handle is None
        self._t2_chunk = chunk
        self._t2_failures = 0
        self.__log_debug('- T2(%s) start', chunk_type(self._t2_chunk))
        self._t2_handle = self._loop.call_later(self._rto, self._t2_expired)

    @no_type_check
    def _t3_expired(self) -> None:
        if False:
            i = 10
            return i + 15
        self._t3_handle = None
        self.__log_debug('x T3 expired')
        for chunk in self._sent_queue:
            if not self._maybe_abandon(chunk):
                chunk._retransmit = True
        self._update_advanced_peer_ack_point()
        self._fast_recovery_exit = None
        self._flight_size = 0
        self._partial_bytes_acked = 0
        self._ssthresh = max(self._cwnd // 2, 4 * USERDATA_MAX_LENGTH)
        self._cwnd = USERDATA_MAX_LENGTH
        asyncio.ensure_future(self._transmit())

    def _t3_restart(self) -> None:
        if False:
            print('Hello World!')
        self.__log_debug('- T3 restart')
        if self._t3_handle is not None:
            self._t3_handle.cancel()
            self._t3_handle = None
        self._t3_handle = self._loop.call_later(self._rto, self._t3_expired)

    def _t3_start(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert self._t3_handle is None
        self.__log_debug('- T3 start')
        self._t3_handle = self._loop.call_later(self._rto, self._t3_expired)

    def _t3_cancel(self) -> None:
        if False:
            return 10
        if self._t3_handle is not None:
            self.__log_debug('- T3 cancel')
            self._t3_handle.cancel()
            self._t3_handle = None

    @no_type_check
    async def _transmit(self) -> None:
        """
        Transmit outbound data.
        """
        if self._forward_tsn_chunk is not None:
            await self._send_chunk(self._forward_tsn_chunk)
            self._forward_tsn_chunk = None
            if not self._t3_handle:
                self._t3_start()
        if self._fast_recovery_exit is not None:
            burst_size = 2 * USERDATA_MAX_LENGTH
        else:
            burst_size = 4 * USERDATA_MAX_LENGTH
        cwnd = min(self._flight_size + burst_size, self._cwnd)
        retransmit_earliest = True
        for chunk in self._sent_queue:
            if chunk._retransmit:
                if self._fast_recovery_transmit:
                    self._fast_recovery_transmit = False
                elif self._flight_size >= cwnd:
                    return
                self._flight_size_increase(chunk)
                chunk._misses = 0
                chunk._retransmit = False
                chunk._sent_count += 1
                await self._send_chunk(chunk)
                if retransmit_earliest:
                    self._t3_restart()
            retransmit_earliest = False
        while self._outbound_queue and self._flight_size < cwnd:
            chunk = self._outbound_queue.popleft()
            self._sent_queue.append(chunk)
            self._flight_size_increase(chunk)
            chunk._sent_count += 1
            chunk._sent_time = time.time()
            await self._send_chunk(chunk)
            if not self._t3_handle:
                self._t3_start()

    async def _transmit_reconfig(self):
        if self._association_state == self.State.ESTABLISHED and self._reconfig_queue and (not self._reconfig_request):
            streams = self._reconfig_queue[0:RECONFIG_MAX_STREAMS]
            self._reconfig_queue = self._reconfig_queue[RECONFIG_MAX_STREAMS:]
            param = StreamResetOutgoingParam(request_sequence=self._reconfig_request_seq, response_sequence=self._reconfig_response_seq, last_tsn=tsn_minus_one(self._local_tsn), streams=streams)
            self._reconfig_request = param
            self._reconfig_request_seq = tsn_plus_one(self._reconfig_request_seq)
            await self._send_reconfig_param(param)

    @no_type_check
    def _update_advanced_peer_ack_point(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Try to advance "Advanced.Peer.Ack.Point" according to RFC 3758.\n        '
        if uint32_gt(self._last_sacked_tsn, self._advanced_peer_ack_tsn):
            self._advanced_peer_ack_tsn = self._last_sacked_tsn
        done = 0
        streams = {}
        while self._sent_queue and self._sent_queue[0]._abandoned:
            chunk = self._sent_queue.popleft()
            self._advanced_peer_ack_tsn = chunk.tsn
            done += 1
            if not chunk.flags & SCTP_DATA_UNORDERED:
                streams[chunk.stream_id] = chunk.stream_seq
        if done:
            self._forward_tsn_chunk = ForwardTsnChunk()
            self._forward_tsn_chunk.cumulative_tsn = self._advanced_peer_ack_tsn
            self._forward_tsn_chunk.streams = list(streams.items())

    def _update_rto(self, R: float) -> None:
        if False:
            while True:
                i = 10
        '\n        Update RTO given a new roundtrip measurement R.\n        '
        if self._srtt is None:
            self._rttvar = R / 2
            self._srtt = R
        else:
            self._rttvar = (1 - SCTP_RTO_BETA) * self._rttvar + SCTP_RTO_BETA * abs(self._srtt - R)
            self._srtt = (1 - SCTP_RTO_ALPHA) * self._srtt + SCTP_RTO_ALPHA * R
        self._rto = max(SCTP_RTO_MIN, min(self._srtt + 4 * self._rttvar, SCTP_RTO_MAX))

    def _data_channel_close(self, channel, transmit=True):
        if False:
            return 10
        '\n        Request closing the datachannel by sending an Outgoing Stream Reset Request.\n        '
        if channel.readyState not in ['closing', 'closed']:
            channel._setReadyState('closing')
            if self._association_state == self.State.ESTABLISHED:
                self._reconfig_queue.append(channel.id)
                if len(self._reconfig_queue) == 1:
                    asyncio.ensure_future(self._transmit_reconfig())
            else:
                new_queue = deque()
                for queue_item in self._data_channel_queue:
                    if queue_item[0] != channel:
                        new_queue.append(queue_item)
                self._data_channel_queue = new_queue
                if channel.id is not None:
                    self._data_channels.pop(channel.id)
                channel._setReadyState('closed')

    def _data_channel_closed(self, stream_id: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        channel = self._data_channels.pop(stream_id)
        channel._setReadyState('closed')

    async def _data_channel_flush(self) -> None:
        """
        Try to flush buffered data to the SCTP layer.

        We wait until the association is established, as we need to know
        whether we are a client or a server to correctly assign an odd/even ID
        to the data channels.
        """
        if self._association_state != self.State.ESTABLISHED:
            return
        while self._data_channel_queue and (not self._outbound_queue):
            (channel, protocol, user_data) = self._data_channel_queue.popleft()
            stream_id = channel.id
            if stream_id is None:
                stream_id = self._data_channel_id
                while stream_id in self._data_channels:
                    stream_id += 2
                self._data_channels[stream_id] = channel
                channel._setId(stream_id)
            if protocol == WEBRTC_DCEP:
                await self._send(stream_id, protocol, user_data)
            else:
                if channel.maxPacketLifeTime:
                    expiry = time.time() + channel.maxPacketLifeTime / 1000
                else:
                    expiry = None
                await self._send(stream_id, protocol, user_data, expiry=expiry, max_retransmits=channel.maxRetransmits, ordered=channel.ordered)
                channel._addBufferedAmount(-len(user_data))

    def _data_channel_add_negotiated(self, channel: RTCDataChannel) -> None:
        if False:
            return 10
        if channel.id in self._data_channels:
            raise ValueError(f'Data channel with ID {channel.id} already registered')
        self._data_channels[channel.id] = channel
        if self._association_state == self.State.ESTABLISHED:
            channel._setReadyState('open')

    def _data_channel_open(self, channel: RTCDataChannel) -> None:
        if False:
            while True:
                i = 10
        if channel.id is not None:
            if channel.id in self._data_channels:
                raise ValueError(f'Data channel with ID {channel.id} already registered')
            else:
                self._data_channels[channel.id] = channel
        channel_type = DATA_CHANNEL_RELIABLE
        priority = 0
        reliability = 0
        if not channel.ordered:
            channel_type |= 128
        if channel.maxRetransmits is not None:
            channel_type |= 1
            reliability = channel.maxRetransmits
        elif channel.maxPacketLifeTime is not None:
            channel_type |= 2
            reliability = channel.maxPacketLifeTime
        data = pack('!BBHLHH', DATA_CHANNEL_OPEN, channel_type, priority, reliability, len(channel.label), len(channel.protocol))
        data += channel.label.encode('utf8')
        data += channel.protocol.encode('utf8')
        self._data_channel_queue.append((channel, WEBRTC_DCEP, data))
        asyncio.ensure_future(self._data_channel_flush())

    async def _data_channel_receive(self, stream_id: int, pp_id: int, data: bytes) -> None:
        if pp_id == WEBRTC_DCEP and len(data):
            msg_type = data[0]
            if msg_type == DATA_CHANNEL_OPEN and len(data) >= 12:
                assert stream_id not in self._data_channels
                (msg_type, channel_type, priority, reliability, label_length, protocol_length) = unpack_from('!BBHLHH', data)
                pos = 12
                label = data[pos:pos + label_length].decode('utf8')
                pos += label_length
                protocol = data[pos:pos + protocol_length].decode('utf8')
                maxPacketLifeTime = None
                maxRetransmits = None
                if channel_type & 3 == 1:
                    maxRetransmits = reliability
                elif channel_type & 3 == 2:
                    maxPacketLifeTime = reliability
                parameters = RTCDataChannelParameters(label=label, ordered=channel_type & 128 == 0, maxPacketLifeTime=maxPacketLifeTime, maxRetransmits=maxRetransmits, protocol=protocol, id=stream_id)
                channel = RTCDataChannel(self, parameters, False)
                channel._setReadyState('open')
                self._data_channels[stream_id] = channel
                self._data_channel_queue.append((channel, WEBRTC_DCEP, pack('!B', DATA_CHANNEL_ACK)))
                await self._data_channel_flush()
                self.emit('datachannel', channel)
            elif msg_type == DATA_CHANNEL_ACK:
                assert stream_id in self._data_channels
                channel = self._data_channels[stream_id]
                channel._setReadyState('open')
        elif pp_id == WEBRTC_STRING and stream_id in self._data_channels:
            self._data_channels[stream_id].emit('message', data.decode('utf8'))
        elif pp_id == WEBRTC_STRING_EMPTY and stream_id in self._data_channels:
            self._data_channels[stream_id].emit('message', '')
        elif pp_id == WEBRTC_BINARY and stream_id in self._data_channels:
            self._data_channels[stream_id].emit('message', data)
        elif pp_id == WEBRTC_BINARY_EMPTY and stream_id in self._data_channels:
            self._data_channels[stream_id].emit('message', b'')

    def _data_channel_send(self, channel: RTCDataChannel, data: bytes) -> None:
        if False:
            while True:
                i = 10
        if data == '':
            (pp_id, user_data) = (WEBRTC_STRING_EMPTY, b'\x00')
        elif isinstance(data, str):
            (pp_id, user_data) = (WEBRTC_STRING, data.encode('utf8'))
        elif data == b'':
            (pp_id, user_data) = (WEBRTC_BINARY_EMPTY, b'\x00')
        else:
            (pp_id, user_data) = (WEBRTC_BINARY, data)
        channel._addBufferedAmount(len(user_data))
        self._data_channel_queue.append((channel, pp_id, user_data))
        asyncio.ensure_future(self._data_channel_flush())

    class State(enum.Enum):
        CLOSED = 1
        COOKIE_WAIT = 2
        COOKIE_ECHOED = 3
        ESTABLISHED = 4
        SHUTDOWN_PENDING = 5
        SHUTDOWN_SENT = 6
        SHUTDOWN_RECEIVED = 7
        SHUTDOWN_ACK_SENT = 8