import asyncio
import struct
from time import perf_counter
import logging
from typing import Optional, Tuple, NamedTuple
from lbry.utils import LRUCache, is_valid_public_ipv4
from lbry.schema.attrs import country_str_to_int, country_int_to_str
log = logging.getLogger(__name__)
_MAGIC = 1446058291
_PAD_BYTES = b'\x00' * 64
PROTOCOL_VERSION = 1

class SPVPing(NamedTuple):
    magic: int
    protocol_version: int
    pad_bytes: bytes

    def encode(self):
        if False:
            print('Hello World!')
        return struct.pack(b'!lB64s', *self)

    @staticmethod
    def make() -> bytes:
        if False:
            i = 10
            return i + 15
        return SPVPing(_MAGIC, PROTOCOL_VERSION, _PAD_BYTES).encode()

    @classmethod
    def decode(cls, packet: bytes):
        if False:
            i = 10
            return i + 15
        decoded = cls(*struct.unpack(b'!lB64s', packet[:69]))
        if decoded.magic != _MAGIC:
            raise ValueError('invalid magic bytes')
        return decoded
PONG_ENCODING = b'!BBL32s4sH'

class SPVPong(NamedTuple):
    protocol_version: int
    flags: int
    height: int
    tip: bytes
    source_address_raw: bytes
    country: int

    def encode(self):
        if False:
            return 10
        return struct.pack(PONG_ENCODING, *self)

    @staticmethod
    def encode_address(address: str):
        if False:
            i = 10
            return i + 15
        return bytes((int(b) for b in address.split('.')))

    @classmethod
    def make(cls, flags: int, height: int, tip: bytes, source_address: str, country: str) -> bytes:
        if False:
            i = 10
            return i + 15
        return SPVPong(PROTOCOL_VERSION, flags, height, tip, cls.encode_address(source_address), country_str_to_int(country)).encode()

    @classmethod
    def make_sans_source_address(cls, flags: int, height: int, tip: bytes, country: str) -> Tuple[bytes, bytes]:
        if False:
            return 10
        pong = cls.make(flags, height, tip, '0.0.0.0', country)
        return (pong[:38], pong[42:])

    @classmethod
    def decode(cls, packet: bytes):
        if False:
            i = 10
            return i + 15
        return cls(*struct.unpack(PONG_ENCODING, packet[:44]))

    @property
    def available(self) -> bool:
        if False:
            print('Hello World!')
        return self.flags & 1 > 0

    @property
    def ip_address(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '.'.join(map(str, self.source_address_raw))

    @property
    def country_name(self):
        if False:
            return 10
        return country_int_to_str(self.country)

    def __repr__(self) -> str:
        if False:
            return 10
        return f"SPVPong(external_ip={self.ip_address}, version={self.protocol_version}, available={('True' if self.flags & 1 > 0 else 'False')}, height={self.height}, tip={self.tip[::-1].hex()}, country={self.country_name})"

class SPVServerStatusProtocol(asyncio.DatagramProtocol):

    def __init__(self, height: int, tip: bytes, country: str, throttle_cache_size: int=1024, throttle_reqs_per_sec: int=10, allow_localhost: bool=False, allow_lan: bool=False):
        if False:
            while True:
                i = 10
        super().__init__()
        self.transport: Optional[asyncio.transports.DatagramTransport] = None
        self._height = height
        self._tip = tip
        self._flags = 0
        self._country = country
        self._left_cache = self._right_cache = None
        self.update_cached_response()
        self._throttle = LRUCache(throttle_cache_size)
        self._should_log = LRUCache(throttle_cache_size)
        self._min_delay = 1 / throttle_reqs_per_sec
        self._allow_localhost = allow_localhost
        self._allow_lan = allow_lan
        self.closed = asyncio.Event()

    def update_cached_response(self):
        if False:
            return 10
        (self._left_cache, self._right_cache) = SPVPong.make_sans_source_address(self._flags, max(0, self._height), self._tip, self._country)

    def set_unavailable(self):
        if False:
            i = 10
            return i + 15
        self._flags &= 254
        self.update_cached_response()

    def set_available(self):
        if False:
            for i in range(10):
                print('nop')
        self._flags |= 1
        self.update_cached_response()

    def set_height(self, height: int, tip: bytes):
        if False:
            print('Hello World!')
        (self._height, self._tip) = (height, tip)
        self.update_cached_response()

    def should_throttle(self, host: str):
        if False:
            i = 10
            return i + 15
        now = perf_counter()
        last_requested = self._throttle.get(host, default=0)
        self._throttle[host] = now
        if now - last_requested < self._min_delay:
            log_cnt = self._should_log.get(host, default=0) + 1
            if log_cnt % 100 == 0:
                log.warning('throttle spv status to %s', host)
            self._should_log[host] = log_cnt
            return True
        return False

    def make_pong(self, host):
        if False:
            print('Hello World!')
        return self._left_cache + SPVPong.encode_address(host) + self._right_cache

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        if False:
            for i in range(10):
                print('nop')
        if self.should_throttle(addr[0]):
            return
        try:
            SPVPing.decode(data)
        except (ValueError, struct.error, AttributeError, TypeError):
            return
        if addr[1] >= 1024 and is_valid_public_ipv4(addr[0], allow_localhost=self._allow_localhost, allow_lan=self._allow_lan):
            self.transport.sendto(self.make_pong(addr[0]), addr)
        else:
            log.warning('odd packet from %s:%i', addr[0], addr[1])

    def connection_made(self, transport) -> None:
        if False:
            print('Hello World!')
        self.transport = transport
        self.closed.clear()

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.transport = None
        self.closed.set()

    async def close(self):
        if self.transport:
            self.transport.close()
        await self.closed.wait()

class StatusServer:

    def __init__(self):
        if False:
            return 10
        self._protocol: Optional[SPVServerStatusProtocol] = None

    async def start(self, height: int, tip: bytes, country: str, interface: str, port: int, allow_lan: bool=False):
        if self.is_running:
            return
        loop = asyncio.get_event_loop()
        interface = interface if interface.lower() != 'localhost' else '127.0.0.1'
        self._protocol = SPVServerStatusProtocol(height, tip, country, allow_localhost=interface == '127.0.0.1', allow_lan=allow_lan)
        await loop.create_datagram_endpoint(lambda : self._protocol, (interface, port))
        log.info('started udp status server on %s:%i', interface, port)

    async def stop(self):
        if self.is_running:
            await self._protocol.close()
            self._protocol = None

    @property
    def is_running(self):
        if False:
            while True:
                i = 10
        return self._protocol is not None

    def set_unavailable(self):
        if False:
            while True:
                i = 10
        if self.is_running:
            self._protocol.set_unavailable()

    def set_available(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_running:
            self._protocol.set_available()

    def set_height(self, height: int, tip: bytes):
        if False:
            i = 10
            return i + 15
        if self.is_running:
            self._protocol.set_height(height, tip)

class SPVStatusClientProtocol(asyncio.DatagramProtocol):

    def __init__(self, responses: asyncio.Queue):
        if False:
            while True:
                i = 10
        super().__init__()
        self.transport: Optional[asyncio.transports.DatagramTransport] = None
        self.responses = responses
        self._ping_packet = SPVPing.make()

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        if False:
            while True:
                i = 10
        try:
            self.responses.put_nowait(((addr, perf_counter()), SPVPong.decode(data)))
        except (ValueError, struct.error, AttributeError, TypeError, RuntimeError):
            return

    def connection_made(self, transport) -> None:
        if False:
            while True:
                i = 10
        self.transport = transport

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if False:
            while True:
                i = 10
        self.transport = None
        log.info('closed udp spv server selection client')

    def ping(self, server: Tuple[str, int]):
        if False:
            i = 10
            return i + 15
        self.transport.sendto(self._ping_packet, server)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if self.transport:
            self.transport.close()