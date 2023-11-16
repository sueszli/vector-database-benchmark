from __future__ import annotations
import contextlib
import enum
import errno
import hmac
import os
import struct
import warnings
import weakref
from itertools import count
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Generic, Iterable, Iterator, TypeVar, Union
from weakref import ReferenceType, WeakValueDictionary
import attr
import trio
from ._util import NoPublicConstructor, final
if TYPE_CHECKING:
    from types import TracebackType
    from OpenSSL import SSL
    from OpenSSL.SSL import Context
    from typing_extensions import Self, TypeAlias
    from trio.socket import SocketType
MAX_UDP_PACKET_SIZE = 65527

def packet_header_overhead(sock: SocketType) -> int:
    if False:
        print('Hello World!')
    if sock.family == trio.socket.AF_INET:
        return 28
    else:
        return 48

def worst_case_mtu(sock: SocketType) -> int:
    if False:
        while True:
            i = 10
    if sock.family == trio.socket.AF_INET:
        return 576 - packet_header_overhead(sock)
    else:
        return 1280 - packet_header_overhead(sock)

def best_guess_mtu(sock: SocketType) -> int:
    if False:
        while True:
            i = 10
    return 1500 - packet_header_overhead(sock)

class ContentType(enum.IntEnum):
    change_cipher_spec = 20
    alert = 21
    handshake = 22
    application_data = 23
    heartbeat = 24

class HandshakeType(enum.IntEnum):
    hello_request = 0
    client_hello = 1
    server_hello = 2
    hello_verify_request = 3
    new_session_ticket = 4
    end_of_early_data = 4
    encrypted_extensions = 8
    certificate = 11
    server_key_exchange = 12
    certificate_request = 13
    server_hello_done = 14
    certificate_verify = 15
    client_key_exchange = 16
    finished = 20
    certificate_url = 21
    certificate_status = 22
    supplemental_data = 23
    key_update = 24
    compressed_certificate = 25
    ekt_key = 26
    message_hash = 254

class ProtocolVersion:
    DTLS10 = bytes([254, 255])
    DTLS12 = bytes([254, 253])
EPOCH_MASK = 65535 << 6 * 8

class BadPacket(Exception):
    pass

def part_of_handshake_untrusted(packet: bytes) -> bool:
    if False:
        i = 10
        return i + 15
    return packet[3:5] == b'\x00\x00'

def is_client_hello_untrusted(packet: bytes) -> bool:
    if False:
        i = 10
        return i + 15
    try:
        return packet[0] == ContentType.handshake and packet[13] == HandshakeType.client_hello
    except IndexError:
        return False
RECORD_HEADER = struct.Struct('!B2sQH')

def to_hex(data: bytes) -> str:
    if False:
        print('Hello World!')
    return data.hex()

@attr.frozen
class Record:
    content_type: int
    version: bytes = attr.ib(repr=to_hex)
    epoch_seqno: int
    payload: bytes = attr.ib(repr=to_hex)

def records_untrusted(packet: bytes) -> Iterator[Record]:
    if False:
        i = 10
        return i + 15
    i = 0
    while i < len(packet):
        try:
            (ct, version, epoch_seqno, payload_len) = RECORD_HEADER.unpack_from(packet, i)
        except struct.error as exc:
            raise BadPacket('invalid record header') from exc
        i += RECORD_HEADER.size
        payload = packet[i:i + payload_len]
        if len(payload) != payload_len:
            raise BadPacket('short record')
        i += payload_len
        yield Record(ct, version, epoch_seqno, payload)

def encode_record(record: Record) -> bytes:
    if False:
        return 10
    header = RECORD_HEADER.pack(record.content_type, record.version, record.epoch_seqno, len(record.payload))
    return header + record.payload
HANDSHAKE_MESSAGE_HEADER = struct.Struct('!B3sH3s3s')

@attr.frozen
class HandshakeFragment:
    msg_type: int
    msg_len: int
    msg_seq: int
    frag_offset: int
    frag_len: int
    frag: bytes = attr.ib(repr=to_hex)

def decode_handshake_fragment_untrusted(payload: bytes) -> HandshakeFragment:
    if False:
        return 10
    try:
        (msg_type, msg_len_bytes, msg_seq, frag_offset_bytes, frag_len_bytes) = HANDSHAKE_MESSAGE_HEADER.unpack_from(payload)
    except struct.error as exc:
        raise BadPacket('bad handshake message header') from exc
    msg_len = int.from_bytes(msg_len_bytes, 'big')
    frag_offset = int.from_bytes(frag_offset_bytes, 'big')
    frag_len = int.from_bytes(frag_len_bytes, 'big')
    frag = payload[HANDSHAKE_MESSAGE_HEADER.size:]
    if len(frag) != frag_len:
        raise BadPacket("handshake fragment length doesn't match record length")
    return HandshakeFragment(msg_type, msg_len, msg_seq, frag_offset, frag_len, frag)

def encode_handshake_fragment(hsf: HandshakeFragment) -> bytes:
    if False:
        i = 10
        return i + 15
    hs_header = HANDSHAKE_MESSAGE_HEADER.pack(hsf.msg_type, hsf.msg_len.to_bytes(3, 'big'), hsf.msg_seq, hsf.frag_offset.to_bytes(3, 'big'), hsf.frag_len.to_bytes(3, 'big'))
    return hs_header + hsf.frag

def decode_client_hello_untrusted(packet: bytes) -> tuple[int, bytes, bytes]:
    if False:
        return 10
    try:
        record = next(records_untrusted(packet))
        if record.content_type != ContentType.handshake:
            raise BadPacket('not a handshake record')
        fragment = decode_handshake_fragment_untrusted(record.payload)
        if fragment.msg_type != HandshakeType.client_hello:
            raise BadPacket('not a ClientHello')
        if fragment.frag_offset != 0:
            raise BadPacket('fragmented ClientHello')
        if fragment.frag_len != fragment.msg_len:
            raise BadPacket('fragmented ClientHello')
        body = fragment.frag
        session_id_len = body[2 + 32]
        cookie_len_offset = 2 + 32 + 1 + session_id_len
        cookie_len = body[cookie_len_offset]
        cookie_start = cookie_len_offset + 1
        cookie_end = cookie_start + cookie_len
        before_cookie = body[:cookie_len_offset]
        cookie = body[cookie_start:cookie_end]
        after_cookie = body[cookie_end:]
        if len(cookie) != cookie_len:
            raise BadPacket('short cookie')
        return (record.epoch_seqno, cookie, before_cookie + after_cookie)
    except (struct.error, IndexError) as exc:
        raise BadPacket('bad ClientHello') from exc

@attr.frozen
class HandshakeMessage:
    record_version: bytes = attr.ib(repr=to_hex)
    msg_type: HandshakeType
    msg_seq: int
    body: bytearray = attr.ib(repr=to_hex)

@attr.frozen
class PseudoHandshakeMessage:
    record_version: bytes = attr.ib(repr=to_hex)
    content_type: int
    payload: bytes = attr.ib(repr=to_hex)

@attr.frozen
class OpaqueHandshakeMessage:
    record: Record
_AnyHandshakeMessage: TypeAlias = Union[HandshakeMessage, PseudoHandshakeMessage, OpaqueHandshakeMessage]

def decode_volley_trusted(volley: bytes) -> list[_AnyHandshakeMessage]:
    if False:
        print('Hello World!')
    messages: list[_AnyHandshakeMessage] = []
    messages_by_seq = {}
    for record in records_untrusted(volley):
        if record.epoch_seqno & EPOCH_MASK:
            messages.append(OpaqueHandshakeMessage(record))
        elif record.content_type in (ContentType.change_cipher_spec, ContentType.alert):
            messages.append(PseudoHandshakeMessage(record.version, record.content_type, record.payload))
        else:
            assert record.content_type == ContentType.handshake
            fragment = decode_handshake_fragment_untrusted(record.payload)
            msg_type = HandshakeType(fragment.msg_type)
            if fragment.msg_seq not in messages_by_seq:
                msg = HandshakeMessage(record.version, msg_type, fragment.msg_seq, bytearray(fragment.msg_len))
                messages.append(msg)
                messages_by_seq[fragment.msg_seq] = msg
            else:
                msg = messages_by_seq[fragment.msg_seq]
            assert msg.msg_type == fragment.msg_type
            assert msg.msg_seq == fragment.msg_seq
            assert len(msg.body) == fragment.msg_len
            msg.body[fragment.frag_offset:fragment.frag_offset + fragment.frag_len] = fragment.frag
    return messages

class RecordEncoder:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self._record_seq = count()

    def set_first_record_number(self, n: int) -> None:
        if False:
            return 10
        self._record_seq = count(n)

    def encode_volley(self, messages: Iterable[_AnyHandshakeMessage], mtu: int) -> list[bytearray]:
        if False:
            i = 10
            return i + 15
        packets = []
        packet = bytearray()
        for message in messages:
            if isinstance(message, OpaqueHandshakeMessage):
                encoded = encode_record(message.record)
                if mtu - len(packet) - len(encoded) <= 0:
                    packets.append(packet)
                    packet = bytearray()
                packet += encoded
                assert len(packet) <= mtu
            elif isinstance(message, PseudoHandshakeMessage):
                space = mtu - len(packet) - RECORD_HEADER.size - len(message.payload)
                if space <= 0:
                    packets.append(packet)
                    packet = bytearray()
                packet += RECORD_HEADER.pack(message.content_type, message.record_version, next(self._record_seq), len(message.payload))
                packet += message.payload
                assert len(packet) <= mtu
            else:
                msg_len_bytes = len(message.body).to_bytes(3, 'big')
                frag_offset = 0
                frags_encoded = 0
                while frag_offset < len(message.body) or not frags_encoded:
                    space = mtu - len(packet) - RECORD_HEADER.size - HANDSHAKE_MESSAGE_HEADER.size
                    if space <= 0:
                        packets.append(packet)
                        packet = bytearray()
                        continue
                    frag = message.body[frag_offset:frag_offset + space]
                    frag_offset_bytes = frag_offset.to_bytes(3, 'big')
                    frag_len_bytes = len(frag).to_bytes(3, 'big')
                    frag_offset += len(frag)
                    packet += RECORD_HEADER.pack(ContentType.handshake, message.record_version, next(self._record_seq), HANDSHAKE_MESSAGE_HEADER.size + len(frag))
                    packet += HANDSHAKE_MESSAGE_HEADER.pack(message.msg_type, msg_len_bytes, message.msg_seq, frag_offset_bytes, frag_len_bytes)
                    packet += frag
                    frags_encoded += 1
                    assert len(packet) <= mtu
        if packet:
            packets.append(packet)
        return packets
COOKIE_REFRESH_INTERVAL = 30
KEY_BYTES = 32
COOKIE_HASH = 'sha256'
SALT_BYTES = 8
COOKIE_LENGTH = 32

def _current_cookie_tick() -> int:
    if False:
        return 10
    return int(trio.current_time() / COOKIE_REFRESH_INTERVAL)

def _signable(*fields: bytes) -> bytes:
    if False:
        return 10
    out = []
    for field in fields:
        out.append(struct.pack('!Q', len(field)))
        out.append(field)
    return b''.join(out)

def _make_cookie(key: bytes, salt: bytes, tick: int, address: Any, client_hello_bits: bytes) -> bytes:
    if False:
        return 10
    assert len(salt) == SALT_BYTES
    assert len(key) == KEY_BYTES
    signable_data = _signable(salt, struct.pack('!Q', tick), _signable(*(str(part).encode() for part in address)), client_hello_bits)
    return (salt + hmac.digest(key, signable_data, COOKIE_HASH))[:COOKIE_LENGTH]

def valid_cookie(key: bytes, cookie: bytes, address: Any, client_hello_bits: bytes) -> bool:
    if False:
        print('Hello World!')
    if len(cookie) > SALT_BYTES:
        salt = cookie[:SALT_BYTES]
        tick = _current_cookie_tick()
        cur_cookie = _make_cookie(key, salt, tick, address, client_hello_bits)
        old_cookie = _make_cookie(key, salt, max(tick - 1, 0), address, client_hello_bits)
        return hmac.compare_digest(cookie, cur_cookie) | hmac.compare_digest(cookie, old_cookie)
    else:
        return False

def challenge_for(key: bytes, address: Any, epoch_seqno: int, client_hello_bits: bytes) -> bytes:
    if False:
        print('Hello World!')
    salt = os.urandom(SALT_BYTES)
    tick = _current_cookie_tick()
    cookie = _make_cookie(key, salt, tick, address, client_hello_bits)
    body = ProtocolVersion.DTLS10 + bytes([len(cookie)]) + cookie
    hs = HandshakeFragment(msg_type=HandshakeType.hello_verify_request, msg_len=len(body), msg_seq=0, frag_offset=0, frag_len=len(body), frag=body)
    payload = encode_handshake_fragment(hs)
    packet = encode_record(Record(ContentType.handshake, ProtocolVersion.DTLS10, epoch_seqno, payload))
    return packet
_T = TypeVar('_T')

class _Queue(Generic[_T]):

    def __init__(self, incoming_packets_buffer: int | float):
        if False:
            while True:
                i = 10
        (self.s, self.r) = trio.open_memory_channel[_T](incoming_packets_buffer)

def _read_loop(read_fn: Callable[[int], bytes]) -> bytes:
    if False:
        i = 10
        return i + 15
    chunks = []
    while True:
        try:
            chunk = read_fn(2 ** 14)
        except SSL.WantReadError:
            break
        chunks.append(chunk)
    return b''.join(chunks)

async def handle_client_hello_untrusted(endpoint: DTLSEndpoint, address: Any, packet: bytes) -> None:
    if endpoint._listening_context is None:
        return
    try:
        (epoch_seqno, cookie, bits) = decode_client_hello_untrusted(packet)
    except BadPacket:
        return
    if endpoint._listening_key is None:
        endpoint._listening_key = os.urandom(KEY_BYTES)
    if not valid_cookie(endpoint._listening_key, cookie, address, bits):
        challenge_packet = challenge_for(endpoint._listening_key, address, epoch_seqno, bits)
        try:
            async with endpoint._send_lock:
                await endpoint.socket.sendto(challenge_packet, address)
        except (OSError, trio.ClosedResourceError):
            pass
    else:
        stream = DTLSChannel._create(endpoint, address, endpoint._listening_context)
        stream._record_encoder.set_first_record_number(epoch_seqno)
        try:
            stream._ssl.bio_write(packet)
            stream._ssl.DTLSv1_listen()
        except SSL.Error:
            return
        stream._ssl.bio_write(packet)
        old_stream = endpoint._streams.get(address)
        if old_stream is not None:
            if old_stream._client_hello == (cookie, bits):
                return
            else:
                old_stream._set_replaced()
        stream._client_hello = (cookie, bits)
        endpoint._streams[address] = stream
        endpoint._incoming_connections_q.s.send_nowait(stream)

async def dtls_receive_loop(endpoint_ref: ReferenceType[DTLSEndpoint], sock: SocketType) -> None:
    try:
        while True:
            try:
                (packet, address) = await sock.recvfrom(MAX_UDP_PACKET_SIZE)
            except OSError as exc:
                if exc.errno == errno.ECONNRESET:
                    continue
                else:
                    raise
            endpoint = endpoint_ref()
            try:
                if endpoint is None:
                    return
                if is_client_hello_untrusted(packet):
                    await handle_client_hello_untrusted(endpoint, address, packet)
                elif address in endpoint._streams:
                    stream = endpoint._streams[address]
                    if stream._did_handshake and part_of_handshake_untrusted(packet):
                        await stream._resend_final_volley()
                    else:
                        try:
                            stream._q.s.send_nowait(packet)
                        except trio.WouldBlock:
                            stream._packets_dropped_in_trio += 1
                else:
                    pass
            finally:
                del endpoint
    except trio.ClosedResourceError:
        return
    except OSError as exc:
        if exc.errno in (errno.EBADF, errno.ENOTSOCK):
            return
        else:
            raise

@attr.frozen
class DTLSChannelStatistics:
    """Currently this has only one attribute:

    - ``incoming_packets_dropped_in_trio`` (``int``): Gives a count of the number of
      incoming packets from this peer that Trio successfully received from the
      network, but then got dropped because the internal channel buffer was full. If
      this is non-zero, then you might want to call ``receive`` more often, or use a
      larger ``incoming_packets_buffer``, or just not worry about it because your
      UDP-based protocol should be able to handle the occasional lost packet, right?

    """
    incoming_packets_dropped_in_trio: int

@final
class DTLSChannel(trio.abc.Channel[bytes], metaclass=NoPublicConstructor):
    """A DTLS connection.

    This class has no public constructor – you get instances by calling
    `DTLSEndpoint.serve` or `~DTLSEndpoint.connect`.

    .. attribute:: endpoint

       The `DTLSEndpoint` that this connection is using.

    .. attribute:: peer_address

       The IP/port of the remote peer that this connection is associated with.

    """

    def __init__(self, endpoint: DTLSEndpoint, peer_address: Any, ctx: Context):
        if False:
            while True:
                i = 10
        self.endpoint = endpoint
        self.peer_address = peer_address
        self._packets_dropped_in_trio = 0
        self._client_hello = None
        self._did_handshake = False
        ctx.set_options(SSL.OP_NO_QUERY_MTU | SSL.OP_NO_RENEGOTIATION)
        self._ssl = SSL.Connection(ctx)
        self._handshake_mtu = 0
        self.set_ciphertext_mtu(best_guess_mtu(self.endpoint.socket))
        self._replaced = False
        self._closed = False
        self._q = _Queue[bytes](endpoint.incoming_packets_buffer)
        self._handshake_lock = trio.Lock()
        self._record_encoder: RecordEncoder = RecordEncoder()
        self._final_volley: list[_AnyHandshakeMessage] = []

    def _set_replaced(self) -> None:
        if False:
            i = 10
            return i + 15
        self._replaced = True
        self._q.s.close()

    def _check_replaced(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._replaced:
            raise trio.BrokenResourceError('peer tore down this connection to start a new one')

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        "Close this connection.\n\n        `DTLSChannel`\\s don't actually own any OS-level resources – the\n        socket is owned by the `DTLSEndpoint`, not the individual connections. So\n        you don't really *have* to call this. But it will interrupt any other tasks\n        calling `receive` with a `ClosedResourceError`, and cause future attempts to use\n        this connection to fail.\n\n        You can also use this object as a synchronous or asynchronous context manager.\n\n        "
        if self._closed:
            return
        self._closed = True
        if self.endpoint._streams.get(self.peer_address) is self:
            del self.endpoint._streams[self.peer_address]
        self._q.r.close()

    def __enter__(self) -> Self:
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        return self.close()

    async def aclose(self) -> None:
        """Close this connection, but asynchronously.

        This is included to satisfy the `trio.abc.Channel` contract. It's
        identical to `close`, but async.

        """
        self.close()
        await trio.lowlevel.checkpoint()

    async def _send_volley(self, volley_messages: list[_AnyHandshakeMessage]) -> None:
        packets = self._record_encoder.encode_volley(volley_messages, self._handshake_mtu)
        for packet in packets:
            async with self.endpoint._send_lock:
                await self.endpoint.socket.sendto(packet, self.peer_address)

    async def _resend_final_volley(self) -> None:
        await self._send_volley(self._final_volley)

    async def do_handshake(self, *, initial_retransmit_timeout: float=1.0) -> None:
        """Perform the handshake.

        Calling this is optional – if you don't, then it will be automatically called
        the first time you call `send` or `receive`. But calling it explicitly can be
        useful in case you want to control the retransmit timeout, use a cancel scope to
        place an overall timeout on the handshake, or catch errors from the handshake
        specifically.

        It's safe to call this multiple times, or call it simultaneously from multiple
        tasks – the first call will perform the handshake, and the rest will be no-ops.

        Args:

          initial_retransmit_timeout (float): Since UDP is an unreliable protocol, it's
            possible that some of the packets we send during the handshake will get
            lost. To handle this, DTLS uses a timer to automatically retransmit
            handshake packets that don't receive a response. This lets you set the
            timeout we use to detect packet loss. Ideally, it should be set to ~1.5
            times the round-trip time to your peer, but 1 second is a reasonable
            default. There's `some useful guidance here
            <https://tlswg.org/dtls13-spec/draft-ietf-tls-dtls13.html#name-timer-values>`__.

            This is the *initial* timeout, because if packets keep being lost then Trio
            will automatically back off to longer values, to avoid overloading the
            network.

        """
        async with self._handshake_lock:
            if self._did_handshake:
                return
            timeout = initial_retransmit_timeout
            volley_messages: list[_AnyHandshakeMessage] = []
            volley_failed_sends = 0

            def read_volley() -> list[_AnyHandshakeMessage]:
                if False:
                    print('Hello World!')
                volley_bytes = _read_loop(self._ssl.bio_read)
                new_volley_messages = decode_volley_trusted(volley_bytes)
                if new_volley_messages and volley_messages and isinstance(new_volley_messages[0], HandshakeMessage) and isinstance(volley_messages[0], HandshakeMessage) and (new_volley_messages[0].msg_seq == volley_messages[0].msg_seq):
                    return []
                else:
                    return new_volley_messages
            with contextlib.suppress(SSL.WantReadError):
                self._ssl.do_handshake()
            volley_messages = read_volley()
            if not volley_messages:
                raise SSL.Error("something wrong with peer's ClientHello")
            while True:
                assert volley_messages
                self._check_replaced()
                await self._send_volley(volley_messages)
                self.endpoint._ensure_receive_loop()
                with trio.move_on_after(timeout) as cscope:
                    async for packet in self._q.r:
                        self._ssl.bio_write(packet)
                        try:
                            self._ssl.do_handshake()
                        except (SSL.WantReadError, SSL.Error):
                            pass
                        else:
                            self._did_handshake = True
                            self._final_volley = read_volley()
                            await self._send_volley(self._final_volley)
                            return
                        maybe_volley = read_volley()
                        if maybe_volley:
                            if isinstance(maybe_volley[0], PseudoHandshakeMessage) and maybe_volley[0].content_type == ContentType.alert:
                                await self._send_volley(maybe_volley)
                            else:
                                volley_messages = maybe_volley
                                if volley_failed_sends == 0:
                                    timeout = initial_retransmit_timeout
                                volley_failed_sends = 0
                                break
                    else:
                        assert self._replaced
                        self._check_replaced()
                if cscope.cancelled_caught:
                    timeout = min(2 * timeout, 60.0)
                    volley_failed_sends += 1
                    if volley_failed_sends == 2:
                        self._handshake_mtu = min(self._handshake_mtu, worst_case_mtu(self.endpoint.socket))

    async def send(self, data: bytes) -> None:
        """Send a packet of data, securely."""
        if self._closed:
            raise trio.ClosedResourceError
        if not data:
            raise ValueError("openssl doesn't support sending empty DTLS packets")
        if not self._did_handshake:
            await self.do_handshake()
        self._check_replaced()
        self._ssl.write(data)
        async with self.endpoint._send_lock:
            await self.endpoint.socket.sendto(_read_loop(self._ssl.bio_read), self.peer_address)

    async def receive(self) -> bytes:
        """Fetch the next packet of data from this connection's peer, waiting if
        necessary.

        This is safe to call from multiple tasks simultaneously, in case you have some
        reason to do that. And more importantly, it's cancellation-safe, meaning that
        cancelling a call to `receive` will never cause a packet to be lost or corrupt
        the underlying connection.

        """
        if not self._did_handshake:
            await self.do_handshake()
        while True:
            try:
                packet = await self._q.r.receive()
            except trio.EndOfChannel:
                assert self._replaced
                self._check_replaced()
            self._ssl.bio_write(packet)
            cleartext = _read_loop(self._ssl.read)
            if cleartext:
                return cleartext

    def set_ciphertext_mtu(self, new_mtu: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Tells Trio the `largest amount of data that can be sent in a single packet to\n        this peer <https://en.wikipedia.org/wiki/Maximum_transmission_unit>`__.\n\n        Trio doesn't actually enforce this limit – if you pass a huge packet to `send`,\n        then we'll dutifully encrypt it and attempt to send it. But calling this method\n        does have two useful effects:\n\n        - If called before the handshake is performed, then Trio will automatically\n          fragment handshake messages to fit within the given MTU. It also might\n          fragment them even smaller, if it detects signs of packet loss, so setting\n          this should never be necessary to make a successful connection. But, the\n          packet loss detection only happens after multiple timeouts have expired, so if\n          you have reason to believe that a smaller MTU is required, then you can set\n          this to skip those timeouts and establish the connection more quickly.\n\n        - It changes the value returned from `get_cleartext_mtu`. So if you have some\n          kind of estimate of the network-level MTU, then you can use this to figure out\n          how much overhead DTLS will need for hashes/padding/etc., and how much space\n          you have left for your application data.\n\n        The MTU here is measuring the largest UDP *payload* you think can be sent, the\n        amount of encrypted data that can be handed to the operating system in a single\n        call to `send`. It should *not* include IP/UDP headers. Note that OS estimates\n        of the MTU often are link-layer MTUs, so you have to subtract off 28 bytes on\n        IPv4 and 48 bytes on IPv6 to get the ciphertext MTU.\n\n        By default, Trio assumes an MTU of 1472 bytes on IPv4, and 1452 bytes on IPv6,\n        which correspond to the common Ethernet MTU of 1500 bytes after accounting for\n        IP/UDP overhead.\n\n        "
        self._handshake_mtu = new_mtu
        self._ssl.set_ciphertext_mtu(new_mtu)

    def get_cleartext_mtu(self) -> int:
        if False:
            while True:
                i = 10
        'Returns the largest number of bytes that you can pass in a single call to\n        `send` while still fitting within the network-level MTU.\n\n        See `set_ciphertext_mtu` for more details.\n\n        '
        if not self._did_handshake:
            raise trio.NeedHandshakeError
        return self._ssl.get_cleartext_mtu()

    def statistics(self) -> DTLSChannelStatistics:
        if False:
            i = 10
            return i + 15
        'Returns a `DTLSChannelStatistics` object with statistics about this connection.'
        return DTLSChannelStatistics(self._packets_dropped_in_trio)

@final
class DTLSEndpoint:
    """A DTLS endpoint.

    A single UDP socket can handle arbitrarily many DTLS connections simultaneously,
    acting as a client or server as needed. A `DTLSEndpoint` object holds a UDP socket
    and manages these connections, which are represented as `DTLSChannel` objects.

    Args:
      socket: (trio.socket.SocketType): A ``SOCK_DGRAM`` socket. If you want to accept
        incoming connections in server mode, then you should probably bind the socket to
        some known port.
      incoming_packets_buffer (int): Each `DTLSChannel` using this socket has its own
        buffer that holds incoming packets until you call `~DTLSChannel.receive` to read
        them. This lets you adjust the size of this buffer. `~DTLSChannel.statistics`
        lets you check if the buffer has overflowed.

    .. attribute:: socket
                   incoming_packets_buffer

       Both constructor arguments are also exposed as attributes, in case you need to
       access them later.

    """

    def __init__(self, socket: SocketType, *, incoming_packets_buffer: int=10):
        if False:
            return 10
        global SSL
        from OpenSSL import SSL
        self._initialized: bool = False
        if socket.type != trio.socket.SOCK_DGRAM:
            raise ValueError('DTLS requires a SOCK_DGRAM socket')
        self._initialized = True
        self.socket: SocketType = socket
        self.incoming_packets_buffer = incoming_packets_buffer
        self._token = trio.lowlevel.current_trio_token()
        self._streams: WeakValueDictionary[Any, DTLSChannel] = WeakValueDictionary()
        self._listening_context: Context | None = None
        self._listening_key: bytes | None = None
        self._incoming_connections_q = _Queue[DTLSChannel](float('inf'))
        self._send_lock = trio.Lock()
        self._closed = False
        self._receive_loop_spawned = False

    def _ensure_receive_loop(self) -> None:
        if False:
            print('Hello World!')
        if not self._receive_loop_spawned:
            trio.lowlevel.spawn_system_task(dtls_receive_loop, weakref.ref(self), self.socket)
            self._receive_loop_spawned = True

    def __del__(self) -> None:
        if False:
            while True:
                i = 10
        if not self._initialized:
            return
        if not self._closed:
            with contextlib.suppress(RuntimeError):
                self._token.run_sync_soon(self.close)
            warnings.warn(f'unclosed DTLS endpoint {self!r}', ResourceWarning, source=self, stacklevel=1)

    def close(self) -> None:
        if False:
            while True:
                i = 10
        'Close this socket, and all associated DTLS connections.\n\n        This object can also be used as a context manager.\n\n        '
        self._closed = True
        self.socket.close()
        for stream in list(self._streams.values()):
            stream.close()
        self._incoming_connections_q.s.close()

    def __enter__(self) -> Self:
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        if False:
            print('Hello World!')
        return self.close()

    def _check_closed(self) -> None:
        if False:
            while True:
                i = 10
        if self._closed:
            raise trio.ClosedResourceError

    async def serve(self, ssl_context: Context, async_fn: Callable[..., Awaitable[object]], *args: Any, task_status: trio.TaskStatus[None]=trio.TASK_STATUS_IGNORED) -> None:
        """Listen for incoming connections, and spawn a handler for each using an
        internal nursery.

        Similar to `~trio.serve_tcp`, this function never returns until cancelled, or
        the `DTLSEndpoint` is closed and all handlers have exited.

        Usage commonly looks like::

            async def handler(dtls_channel):
                ...

            async with trio.open_nursery() as nursery:
                await nursery.start(dtls_endpoint.serve, ssl_context, handler)
                # ... do other things here ...

        The ``dtls_channel`` passed into the handler function has already performed the
        "cookie exchange" part of the DTLS handshake, so the peer address is
        trustworthy. But the actual cryptographic handshake doesn't happen until you
        start using it, giving you a chance for any last minute configuration, and the
        option to catch and handle handshake errors.

        Args:
          ssl_context (OpenSSL.SSL.Context): The PyOpenSSL context object to use for
            incoming connections.
          async_fn: The handler function that will be invoked for each incoming
            connection.

        """
        self._check_closed()
        if self._listening_context is not None:
            raise trio.BusyResourceError('another task is already listening')
        try:
            self.socket.getsockname()
        except OSError:
            raise RuntimeError('DTLS socket must be bound before it can serve') from None
        self._ensure_receive_loop()
        ssl_context.set_cookie_verify_callback(lambda *_: True)
        try:
            self._listening_context = ssl_context
            task_status.started()

            async def handler_wrapper(stream: DTLSChannel) -> None:
                with stream:
                    await async_fn(stream, *args)
            async with trio.open_nursery() as nursery:
                async for stream in self._incoming_connections_q.r:
                    nursery.start_soon(handler_wrapper, stream)
        finally:
            self._listening_context = None

    def connect(self, address: tuple[str, int], ssl_context: Context) -> DTLSChannel:
        if False:
            print('Hello World!')
        'Initiate an outgoing DTLS connection.\n\n        Notice that this is a synchronous method. That\'s because it doesn\'t actually\n        initiate any I/O – it just sets up a `DTLSChannel` object. The actual handshake\n        doesn\'t occur until you start using the `DTLSChannel`. This gives you a chance\n        to do further configuration first, like setting MTU etc.\n\n        Args:\n          address: The address to connect to. Usually a (host, port) tuple, like\n            ``("127.0.0.1", 12345)``.\n          ssl_context (OpenSSL.SSL.Context): The PyOpenSSL context object to use for\n            this connection.\n\n        Returns:\n          DTLSChannel\n\n        '
        self._check_closed()
        channel = DTLSChannel._create(self, address, ssl_context)
        channel._ssl.set_connect_state()
        old_channel = self._streams.get(address)
        if old_channel is not None:
            old_channel._set_replaced()
        self._streams[address] = channel
        return channel