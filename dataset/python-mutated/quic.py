from __future__ import annotations
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from logging import DEBUG
from logging import ERROR
from logging import WARNING
from ssl import VerifyMode
from aioquic.buffer import Buffer as QuicBuffer
from aioquic.h3.connection import ErrorCode as H3ErrorCode
from aioquic.quic import events as quic_events
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.connection import QuicConnection
from aioquic.quic.connection import QuicConnectionError
from aioquic.quic.connection import QuicConnectionState
from aioquic.quic.connection import QuicErrorCode
from aioquic.quic.connection import stream_is_client_initiated
from aioquic.quic.connection import stream_is_unidirectional
from aioquic.quic.packet import encode_quic_version_negotiation
from aioquic.quic.packet import PACKET_TYPE_INITIAL
from aioquic.quic.packet import pull_quic_header
from aioquic.quic.packet import QuicProtocolVersion
from aioquic.tls import CipherSuite
from aioquic.tls import HandshakeType
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import rsa
from mitmproxy import certs
from mitmproxy import connection
from mitmproxy import ctx
from mitmproxy.net import tls
from mitmproxy.proxy import commands
from mitmproxy.proxy import context
from mitmproxy.proxy import events
from mitmproxy.proxy import layer
from mitmproxy.proxy import tunnel
from mitmproxy.proxy.layers.modes import TransparentProxy
from mitmproxy.proxy.layers.tcp import TCPLayer
from mitmproxy.proxy.layers.tls import TlsClienthelloHook
from mitmproxy.proxy.layers.tls import TlsEstablishedClientHook
from mitmproxy.proxy.layers.tls import TlsEstablishedServerHook
from mitmproxy.proxy.layers.tls import TlsFailedClientHook
from mitmproxy.proxy.layers.tls import TlsFailedServerHook
from mitmproxy.proxy.layers.udp import UDPLayer
from mitmproxy.tls import ClientHello
from mitmproxy.tls import ClientHelloData
from mitmproxy.tls import TlsData

@dataclass
class QuicTlsSettings:
    """
    Settings necessary to establish QUIC's TLS context.
    """
    alpn_protocols: list[str] | None = None
    'A list of supported ALPN protocols.'
    certificate: x509.Certificate | None = None
    'The certificate to use for the connection.'
    certificate_chain: list[x509.Certificate] = field(default_factory=list)
    'A list of additional certificates to send to the peer.'
    certificate_private_key: dsa.DSAPrivateKey | ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey | None = None
    "The certificate's private key."
    cipher_suites: list[CipherSuite] | None = None
    'An optional list of allowed/advertised cipher suites.'
    ca_path: str | None = None
    'An optional path to a directory that contains the necessary information to verify the peer certificate.'
    ca_file: str | None = None
    'An optional path to a PEM file that will be used to verify the peer certificate.'
    verify_mode: VerifyMode | None = None
    "An optional flag that specifies how/if the peer's certificate should be validated."

@dataclass
class QuicTlsData(TlsData):
    """
    Event data for `quic_start_client` and `quic_start_server` event hooks.
    """
    settings: QuicTlsSettings | None = None
    '\n    The associated `QuicTlsSettings` object.\n    This will be set by an addon in the `quic_start_*` event hooks.\n    '

@dataclass
class QuicStartClientHook(commands.StartHook):
    """
    TLS negotiation between mitmproxy and a client over QUIC is about to start.

    An addon is expected to initialize data.settings.
    (by default, this is done by `mitmproxy.addons.tlsconfig`)
    """
    data: QuicTlsData

@dataclass
class QuicStartServerHook(commands.StartHook):
    """
    TLS negotiation between mitmproxy and a server over QUIC is about to start.

    An addon is expected to initialize data.settings.
    (by default, this is done by `mitmproxy.addons.tlsconfig`)
    """
    data: QuicTlsData

@dataclass
class QuicStreamEvent(events.ConnectionEvent):
    """Base class for all QUIC stream events."""
    stream_id: int
    'The ID of the stream the event was fired for.'

@dataclass
class QuicStreamDataReceived(QuicStreamEvent):
    """Event that is fired whenever data is received on a stream."""
    data: bytes
    'The data which was received.'
    end_stream: bool
    'Whether the STREAM frame had the FIN bit set.'

@dataclass
class QuicStreamReset(QuicStreamEvent):
    """Event that is fired when the remote peer resets a stream."""
    error_code: int
    'The error code that triggered the reset.'

class QuicStreamCommand(commands.ConnectionCommand):
    """Base class for all QUIC stream commands."""
    stream_id: int
    'The ID of the stream the command was issued for.'

    def __init__(self, connection: connection.Connection, stream_id: int) -> None:
        if False:
            while True:
                i = 10
        super().__init__(connection)
        self.stream_id = stream_id

class SendQuicStreamData(QuicStreamCommand):
    """Command that sends data on a stream."""
    data: bytes
    'The data which should be sent.'
    end_stream: bool
    'Whether the FIN bit should be set in the STREAM frame.'

    def __init__(self, connection: connection.Connection, stream_id: int, data: bytes, end_stream: bool=False) -> None:
        if False:
            print('Hello World!')
        super().__init__(connection, stream_id)
        self.data = data
        self.end_stream = end_stream

class ResetQuicStream(QuicStreamCommand):
    """Abruptly terminate the sending part of a stream."""
    error_code: int
    'An error code indicating why the stream is being reset.'

    def __init__(self, connection: connection.Connection, stream_id: int, error_code: int) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(connection, stream_id)
        self.error_code = error_code

class StopQuicStream(QuicStreamCommand):
    """Request termination of the receiving part of a stream."""
    error_code: int
    'An error code indicating why the stream is being stopped.'

    def __init__(self, connection: connection.Connection, stream_id: int, error_code: int) -> None:
        if False:
            return 10
        super().__init__(connection, stream_id)
        self.error_code = error_code

class CloseQuicConnection(commands.CloseConnection):
    """Close a QUIC connection."""
    error_code: int
    'The error code which was specified when closing the connection.'
    frame_type: int | None
    'The frame type which caused the connection to be closed, or `None`.'
    reason_phrase: str
    'The human-readable reason for which the connection was closed.'

    def __init__(self, conn: connection.Connection, error_code: int, frame_type: int | None, reason_phrase: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(conn)
        self.error_code = error_code
        self.frame_type = frame_type
        self.reason_phrase = reason_phrase

class QuicConnectionClosed(events.ConnectionClosed):
    """QUIC connection has been closed."""
    error_code: int
    'The error code which was specified when closing the connection.'
    frame_type: int | None
    'The frame type which caused the connection to be closed, or `None`.'
    reason_phrase: str
    'The human-readable reason for which the connection was closed.'

    def __init__(self, conn: connection.Connection, error_code: int, frame_type: int | None, reason_phrase: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(conn)
        self.error_code = error_code
        self.frame_type = frame_type
        self.reason_phrase = reason_phrase

class QuicSecretsLogger:
    logger: tls.MasterSecretLogger

    def __init__(self, logger: tls.MasterSecretLogger) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.logger = logger

    def write(self, s: str) -> int:
        if False:
            i = 10
            return i + 15
        if s[-1:] == '\n':
            s = s[:-1]
        data = s.encode('ascii')
        self.logger(None, data)
        return len(data) + 1

    def flush(self) -> None:
        if False:
            while True:
                i = 10
        pass

def error_code_to_str(error_code: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Returns the corresponding name of the given error code or a string containing its numeric value.'
    try:
        return H3ErrorCode(error_code).name
    except ValueError:
        try:
            return QuicErrorCode(error_code).name
        except ValueError:
            return f'unknown error (0x{error_code:x})'

def is_success_error_code(error_code: int) -> bool:
    if False:
        while True:
            i = 10
    'Returns whether the given error code actually indicates no error.'
    return error_code in (QuicErrorCode.NO_ERROR, H3ErrorCode.H3_NO_ERROR)

def tls_settings_to_configuration(settings: QuicTlsSettings, is_client: bool, server_name: str | None=None) -> QuicConfiguration:
    if False:
        for i in range(10):
            print('nop')
    'Converts `QuicTlsSettings` to `QuicConfiguration`.'
    return QuicConfiguration(alpn_protocols=settings.alpn_protocols, is_client=is_client, secrets_log_file=QuicSecretsLogger(tls.log_master_secret) if tls.log_master_secret is not None else None, server_name=server_name, cafile=settings.ca_file, capath=settings.ca_path, certificate=settings.certificate, certificate_chain=settings.certificate_chain, cipher_suites=settings.cipher_suites, private_key=settings.certificate_private_key, verify_mode=settings.verify_mode, max_datagram_frame_size=65536)

@dataclass
class QuicClientHello(Exception):
    """Helper error only used in `quic_parse_client_hello`."""
    data: bytes

def quic_parse_client_hello(data: bytes) -> ClientHello:
    if False:
        i = 10
        return i + 15
    'Helper function that parses a client hello packet.'
    buffer = QuicBuffer(data=data)
    header = pull_quic_header(buffer, 8)
    if header.packet_type != PACKET_TYPE_INITIAL:
        raise ValueError('Packet is not initial one.')
    quic = QuicConnection(configuration=QuicConfiguration(is_client=False, certificate='', private_key=''), original_destination_connection_id=header.destination_cid)
    _initialize = quic._initialize

    def server_handle_hello_replacement(input_buf: QuicBuffer, initial_buf: QuicBuffer, handshake_buf: QuicBuffer, onertt_buf: QuicBuffer) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert input_buf.pull_uint8() == HandshakeType.CLIENT_HELLO
        length = 0
        for b in input_buf.pull_bytes(3):
            length = length << 8 | b
        offset = input_buf.tell()
        raise QuicClientHello(input_buf.data_slice(offset, offset + length))

    def initialize_replacement(peer_cid: bytes) -> None:
        if False:
            return 10
        try:
            return _initialize(peer_cid)
        finally:
            quic.tls._server_handle_hello = server_handle_hello_replacement
    quic._initialize = initialize_replacement
    try:
        quic.receive_datagram(data, ('0.0.0.0', 0), now=0)
    except QuicClientHello as hello:
        try:
            return ClientHello(hello.data)
        except EOFError as e:
            raise ValueError('Invalid ClientHello data.') from e
    except QuicConnectionError as e:
        raise ValueError(e.reason_phrase) from e
    raise ValueError('No ClientHello returned.')

class QuicStreamNextLayer(layer.NextLayer):
    """`NextLayer` variant that callbacks `QuicStreamLayer` after layer decision."""

    def __init__(self, context: context.Context, stream: QuicStreamLayer, ask_on_start: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(context, ask_on_start)
        self._stream = stream
        self._layer: layer.Layer | None = None

    @property
    def layer(self) -> layer.Layer | None:
        if False:
            i = 10
            return i + 15
        return self._layer

    @layer.setter
    def layer(self, value: layer.Layer | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._layer = value
        if self._layer:
            self._stream.refresh_metadata()

class QuicStreamLayer(layer.Layer):
    """
    Layer for QUIC streams.
    Serves as a marker for NextLayer and keeps track of the connection states.
    """
    client: connection.Client
    'Virtual client connection for this stream. Use this in QuicRawLayer instead of `context.client`.'
    server: connection.Server
    'Virtual server connection for this stream. Use this in QuicRawLayer instead of `context.server`.'
    child_layer: layer.Layer
    "The stream's child layer."

    def __init__(self, context: context.Context, ignore: bool, stream_id: int) -> None:
        if False:
            print('Hello World!')
        self.client = context.client = context.client.copy()
        self.client.transport_protocol = 'tcp'
        self.client.state = connection.ConnectionState.OPEN
        if stream_is_unidirectional(stream_id):
            self.client.state = connection.ConnectionState.CAN_READ if stream_is_client_initiated(stream_id) else connection.ConnectionState.CAN_WRITE
        self._client_stream_id = stream_id
        self.server = context.server = connection.Server(address=context.server.address, transport_protocol='tcp')
        self._server_stream_id: int | None = None
        super().__init__(context)
        self.child_layer = TCPLayer(context, ignore=True) if ignore else QuicStreamNextLayer(context, self)
        self.refresh_metadata()
        self.handle_event = self.child_layer.handle_event
        self._handle_event = self.child_layer._handle_event

    def _handle_event(self, event: events.Event) -> layer.CommandGenerator[None]:
        if False:
            return 10
        raise AssertionError

    def open_server_stream(self, server_stream_id) -> None:
        if False:
            return 10
        assert self._server_stream_id is None
        self._server_stream_id = server_stream_id
        self.server.timestamp_start = time.time()
        self.server.state = (connection.ConnectionState.CAN_WRITE if stream_is_client_initiated(server_stream_id) else connection.ConnectionState.CAN_READ) if stream_is_unidirectional(server_stream_id) else connection.ConnectionState.OPEN
        self.refresh_metadata()

    def refresh_metadata(self) -> None:
        if False:
            while True:
                i = 10
        child_layer: layer.Layer | None = self.child_layer
        while True:
            if isinstance(child_layer, layer.NextLayer):
                child_layer = child_layer.layer
            elif isinstance(child_layer, tunnel.TunnelLayer):
                child_layer = child_layer.child_layer
            else:
                break
        if isinstance(child_layer, (UDPLayer, TCPLayer)) and child_layer.flow:
            child_layer.flow.metadata['quic_is_unidirectional'] = stream_is_unidirectional(self._client_stream_id)
            child_layer.flow.metadata['quic_initiator'] = 'client' if stream_is_client_initiated(self._client_stream_id) else 'server'
            child_layer.flow.metadata['quic_stream_id_client'] = self._client_stream_id
            child_layer.flow.metadata['quic_stream_id_server'] = self._server_stream_id

    def stream_id(self, client: bool) -> int | None:
        if False:
            while True:
                i = 10
        return self._client_stream_id if client else self._server_stream_id

class RawQuicLayer(layer.Layer):
    """
    This layer is responsible for de-multiplexing QUIC streams into an individual layer stack per stream.
    """
    ignore: bool
    'Indicates whether traffic should be routed as-is.'
    datagram_layer: layer.Layer
    "\n    The layer that is handling datagrams over QUIC. It's like a child_layer, but with a forked context.\n    Instead of having a datagram-equivalent for all `QuicStream*` classes, we use `SendData` and `DataReceived` instead.\n    There is also no need for another `NextLayer` marker, as a missing `QuicStreamLayer` implies UDP,\n    and the connection state is the same as the one of the underlying QUIC connection.\n    "
    client_stream_ids: dict[int, QuicStreamLayer]
    'Maps stream IDs from the client connection to stream layers.'
    server_stream_ids: dict[int, QuicStreamLayer]
    'Maps stream IDs from the server connection to stream layers.'
    connections: dict[connection.Connection, layer.Layer]
    'Maps connections to layers.'
    command_sources: dict[commands.Command, layer.Layer]
    'Keeps track of blocking commands and wakeup requests.'
    next_stream_id: list[int]
    'List containing the next stream ID for all four is_unidirectional/is_client combinations.'

    def __init__(self, context: context.Context, ignore: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(context)
        self.ignore = ignore
        self.datagram_layer = UDPLayer(self.context.fork(), ignore=True) if ignore else layer.NextLayer(self.context.fork())
        self.client_stream_ids = {}
        self.server_stream_ids = {}
        self.connections = {context.client: self.datagram_layer, context.server: self.datagram_layer}
        self.command_sources = {}
        self.next_stream_id = [0, 1, 2, 3]

    def _handle_event(self, event: events.Event) -> layer.CommandGenerator[None]:
        if False:
            while True:
                i = 10
        if isinstance(event, events.Start):
            if self.context.server.timestamp_start is None:
                err = (yield commands.OpenConnection(self.context.server))
                if err:
                    yield commands.CloseConnection(self.context.client)
                    self._handle_event = self.done
                    return
            yield from self.event_to_child(self.datagram_layer, event)
        elif isinstance(event, events.CommandCompleted):
            yield from self.event_to_child(self.command_sources.pop(event.command), event)
        elif isinstance(event, events.MessageInjected):
            if event.flow.client_conn in self.connections:
                yield from self.event_to_child(self.connections[event.flow.client_conn], event)
            elif event.flow.server_conn in self.connections:
                yield from self.event_to_child(self.connections[event.flow.server_conn], event)
            else:
                raise AssertionError(f'Flow not associated: {event.flow!r}')
        elif isinstance(event, QuicStreamEvent) and (event.connection is self.context.client or event.connection is self.context.server):
            from_client = event.connection is self.context.client
            stream_ids = self.client_stream_ids if from_client else self.server_stream_ids
            if event.stream_id in stream_ids:
                stream_layer = stream_ids[event.stream_id]
            else:
                assert stream_is_client_initiated(event.stream_id) == from_client
                if from_client:
                    client_stream_id = event.stream_id
                    server_stream_id = None
                else:
                    client_stream_id = self.get_next_available_stream_id(is_client=False, is_unidirectional=stream_is_unidirectional(event.stream_id))
                    server_stream_id = event.stream_id
                stream_layer = QuicStreamLayer(self.context.fork(), self.ignore, client_stream_id)
                self.client_stream_ids[client_stream_id] = stream_layer
                if server_stream_id is not None:
                    stream_layer.open_server_stream(server_stream_id)
                    self.server_stream_ids[server_stream_id] = stream_layer
                self.connections[stream_layer.client] = stream_layer
                self.connections[stream_layer.server] = stream_layer
                yield from self.event_to_child(stream_layer, events.Start())
            conn = stream_layer.client if from_client else stream_layer.server
            if isinstance(event, QuicStreamDataReceived):
                if event.data:
                    yield from self.event_to_child(stream_layer, events.DataReceived(conn, event.data))
                if event.end_stream:
                    yield from self.close_stream_layer(stream_layer, from_client)
            elif isinstance(event, QuicStreamReset):
                for command in self.close_stream_layer(stream_layer, from_client):
                    if isinstance(command, SendQuicStreamData) and command.stream_id == stream_layer.stream_id(not from_client) and command.end_stream and (not command.data):
                        yield ResetQuicStream(command.connection, command.stream_id, event.error_code)
                    else:
                        yield command
            else:
                raise AssertionError(f'Unexpected stream event: {event!r}')
        elif isinstance(event, QuicConnectionClosed) and (event.connection is self.context.client or event.connection is self.context.server):
            from_client = event.connection is self.context.client
            other_conn = self.context.server if from_client else self.context.client
            if other_conn.connected:
                yield CloseQuicConnection(other_conn, event.error_code, event.frame_type, event.reason_phrase)
            else:
                self._handle_event = self.done
            for command in self.event_to_child(self.datagram_layer, event):
                if not isinstance(command, commands.CloseConnection) or command.connection is not other_conn:
                    yield command
            for (conn, child_layer) in self.connections.items():
                if isinstance(child_layer, QuicStreamLayer) and (conn is child_layer.client if from_client else conn is child_layer.server):
                    conn.state &= ~connection.ConnectionState.CAN_WRITE
                    for command in self.close_stream_layer(child_layer, from_client):
                        if not isinstance(command, SendQuicStreamData) or command.data:
                            yield command
        elif isinstance(event, events.ConnectionEvent):
            yield from self.event_to_child(self.connections[event.connection], event)
        else:
            raise AssertionError(f'Unexpected event: {event!r}')

    def close_stream_layer(self, stream_layer: QuicStreamLayer, client: bool) -> layer.CommandGenerator[None]:
        if False:
            return 10
        'Closes the incoming part of a connection.'
        conn = stream_layer.client if client else stream_layer.server
        conn.state &= ~connection.ConnectionState.CAN_READ
        assert conn.timestamp_start is not None
        if conn.timestamp_end is None:
            conn.timestamp_end = time.time()
            yield from self.event_to_child(stream_layer, events.ConnectionClosed(conn))

    def event_to_child(self, child_layer: layer.Layer, event: events.Event) -> layer.CommandGenerator[None]:
        if False:
            while True:
                i = 10
        'Forwards events to child layers and translates commands.'
        for command in child_layer.handle_event(event):
            if isinstance(child_layer, QuicStreamLayer) and isinstance(command, commands.ConnectionCommand) and (command.connection is child_layer.client or command.connection is child_layer.server):
                to_client = command.connection is child_layer.client
                quic_conn = self.context.client if to_client else self.context.server
                stream_id = child_layer.stream_id(to_client)
                if isinstance(command, commands.SendData):
                    assert stream_id is not None
                    if command.connection.state & connection.ConnectionState.CAN_WRITE:
                        yield SendQuicStreamData(quic_conn, stream_id, command.data)
                elif isinstance(command, commands.CloseConnection):
                    assert stream_id is not None
                    if command.connection.state & connection.ConnectionState.CAN_WRITE:
                        command.connection.state &= ~connection.ConnectionState.CAN_WRITE
                        yield SendQuicStreamData(quic_conn, stream_id, b'', end_stream=True)
                    only_close_our_half = isinstance(command, commands.CloseTcpConnection) and command.half_close
                    if not only_close_our_half:
                        if stream_is_client_initiated(stream_id) == to_client or not stream_is_unidirectional(stream_id):
                            yield StopQuicStream(quic_conn, stream_id, QuicErrorCode.NO_ERROR)
                        yield from self.close_stream_layer(child_layer, to_client)
                elif isinstance(command, commands.OpenConnection):
                    assert not to_client
                    assert stream_id is None
                    client_stream_id = child_layer.stream_id(client=True)
                    assert client_stream_id is not None
                    stream_id = self.get_next_available_stream_id(is_client=True, is_unidirectional=stream_is_unidirectional(client_stream_id))
                    child_layer.open_server_stream(stream_id)
                    self.server_stream_ids[stream_id] = child_layer
                    yield from self.event_to_child(child_layer, events.OpenConnectionCompleted(command, None))
                else:
                    raise AssertionError(f'Unexpected stream connection command: {command!r}')
            else:
                if command.blocking or isinstance(command, commands.RequestWakeup):
                    self.command_sources[command] = child_layer
                if isinstance(command, commands.OpenConnection):
                    self.connections[command.connection] = child_layer
                yield command

    def get_next_available_stream_id(self, is_client: bool, is_unidirectional: bool=False) -> int:
        if False:
            for i in range(10):
                print('nop')
        index = int(is_unidirectional) << 1 | int(not is_client)
        stream_id = self.next_stream_id[index]
        self.next_stream_id[index] = stream_id + 4
        return stream_id

    def done(self, _) -> layer.CommandGenerator[None]:
        if False:
            print('Hello World!')
        yield from ()

class QuicLayer(tunnel.TunnelLayer):
    quic: QuicConnection | None = None
    tls: QuicTlsSettings | None = None

    def __init__(self, context: context.Context, conn: connection.Connection, time: Callable[[], float] | None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(context, tunnel_connection=conn, conn=conn)
        self.child_layer = layer.NextLayer(self.context, ask_on_start=True)
        self._time = time or ctx.master.event_loop.time
        self._wakeup_commands: dict[commands.RequestWakeup, float] = dict()
        conn.tls = True

    def _handle_event(self, event: events.Event) -> layer.CommandGenerator[None]:
        if False:
            return 10
        if isinstance(event, events.Wakeup) and event.command in self._wakeup_commands:
            assert self.quic
            timer = self._wakeup_commands.pop(event.command)
            if self.quic._state is not QuicConnectionState.TERMINATED:
                self.quic.handle_timer(now=max(timer, self._time()))
                yield from super()._handle_event(events.DataReceived(self.tunnel_connection, b''))
        else:
            yield from super()._handle_event(event)

    def event_to_child(self, event: events.Event) -> layer.CommandGenerator[None]:
        if False:
            return 10
        yield from super().event_to_child(event)
        if self.quic:
            yield from self.tls_interact()

    def _handle_command(self, command: commands.Command) -> layer.CommandGenerator[None]:
        if False:
            for i in range(10):
                print('nop')
        'Turns stream commands into aioquic connection invocations.'
        if isinstance(command, QuicStreamCommand) and command.connection is self.conn:
            assert self.quic
            if isinstance(command, SendQuicStreamData):
                self.quic.send_stream_data(command.stream_id, command.data, command.end_stream)
            elif isinstance(command, ResetQuicStream):
                self.quic.reset_stream(command.stream_id, command.error_code)
            elif isinstance(command, StopQuicStream):
                if command.stream_id in self.quic._streams:
                    self.quic.stop_stream(command.stream_id, command.error_code)
            else:
                raise AssertionError(f'Unexpected stream command: {command!r}')
        else:
            yield from super()._handle_command(command)

    def start_tls(self, original_destination_connection_id: bytes | None) -> layer.CommandGenerator[None]:
        if False:
            print('Hello World!')
        'Initiates the aioquic connection.'
        assert not self.quic
        assert not self.tls
        tls_data = QuicTlsData(self.conn, self.context)
        if self.conn is self.context.client:
            yield QuicStartClientHook(tls_data)
        else:
            yield QuicStartServerHook(tls_data)
        if not tls_data.settings:
            yield commands.Log(f'No QUIC context was provided, failing connection.', ERROR)
            yield commands.CloseConnection(self.conn)
            return
        configuration = tls_settings_to_configuration(settings=tls_data.settings, is_client=self.conn is self.context.server, server_name=self.conn.sni)
        self.quic = QuicConnection(configuration=configuration, original_destination_connection_id=original_destination_connection_id)
        self.tls = tls_data.settings
        if original_destination_connection_id is None:
            self.quic.connect(self.conn.peername, now=self._time())
            yield from self.tls_interact()

    def tls_interact(self) -> layer.CommandGenerator[None]:
        if False:
            while True:
                i = 10
        'Retrieves all pending outgoing packets from aioquic and sends the data.'
        assert self.quic
        for (data, addr) in self.quic.datagrams_to_send(now=self._time()):
            assert addr == self.conn.peername
            yield commands.SendData(self.tunnel_connection, data)
        timer = self.quic.get_timer()
        if timer is not None and (not any((existing <= timer for existing in self._wakeup_commands.values()))):
            command = commands.RequestWakeup(timer - self._time())
            self._wakeup_commands[command] = timer
            yield command

    def receive_handshake_data(self, data: bytes) -> layer.CommandGenerator[tuple[bool, str | None]]:
        if False:
            while True:
                i = 10
        assert self.quic
        if data:
            self.quic.receive_datagram(data, self.conn.peername, now=self._time())
        while (event := self.quic.next_event()):
            if isinstance(event, quic_events.ConnectionTerminated):
                err = event.reason_phrase or error_code_to_str(event.error_code)
                return (False, err)
            elif isinstance(event, quic_events.HandshakeCompleted):
                all_certs: list[x509.Certificate] = []
                if self.quic.tls._peer_certificate:
                    all_certs.append(self.quic.tls._peer_certificate)
                all_certs.extend(self.quic.tls._peer_certificate_chain)
                self.conn.timestamp_tls_setup = time.time()
                if event.alpn_protocol:
                    self.conn.alpn = event.alpn_protocol.encode('ascii')
                self.conn.certificate_list = [certs.Cert(cert) for cert in all_certs]
                assert self.quic.tls.key_schedule
                self.conn.cipher = self.quic.tls.key_schedule.cipher_suite.name
                self.conn.tls_version = 'QUIC'
                if self.debug:
                    yield commands.Log(f'{self.debug}[quic] tls established: {self.conn}', DEBUG)
                if self.conn is self.context.client:
                    yield TlsEstablishedClientHook(QuicTlsData(self.conn, self.context, settings=self.tls))
                else:
                    yield TlsEstablishedServerHook(QuicTlsData(self.conn, self.context, settings=self.tls))
                yield from self.tls_interact()
                return (True, None)
            elif isinstance(event, (quic_events.ConnectionIdIssued, quic_events.ConnectionIdRetired, quic_events.PingAcknowledged, quic_events.ProtocolNegotiated)):
                pass
            else:
                raise AssertionError(f'Unexpected event: {event!r}')
        yield from self.tls_interact()
        return (False, None)

    def on_handshake_error(self, err: str) -> layer.CommandGenerator[None]:
        if False:
            print('Hello World!')
        self.conn.error = err
        if self.conn is self.context.client:
            yield TlsFailedClientHook(QuicTlsData(self.conn, self.context, settings=self.tls))
        else:
            yield TlsFailedServerHook(QuicTlsData(self.conn, self.context, settings=self.tls))
        yield from super().on_handshake_error(err)

    def receive_data(self, data: bytes) -> layer.CommandGenerator[None]:
        if False:
            return 10
        assert self.quic
        if data:
            self.quic.receive_datagram(data, self.conn.peername, now=self._time())
        while (event := self.quic.next_event()):
            if isinstance(event, quic_events.ConnectionTerminated):
                if self.debug:
                    reason = event.reason_phrase or error_code_to_str(event.error_code)
                    yield commands.Log(f'{self.debug}[quic] close_notify {self.conn} (reason={reason})', DEBUG)
                yield commands.CloseConnection(self.tunnel_connection)
                return
            elif isinstance(event, quic_events.DatagramFrameReceived):
                yield from self.event_to_child(events.DataReceived(self.conn, event.data))
            elif isinstance(event, quic_events.StreamDataReceived):
                yield from self.event_to_child(QuicStreamDataReceived(self.conn, event.stream_id, event.data, event.end_stream))
            elif isinstance(event, quic_events.StreamReset):
                yield from self.event_to_child(QuicStreamReset(self.conn, event.stream_id, event.error_code))
            elif isinstance(event, (quic_events.ConnectionIdIssued, quic_events.ConnectionIdRetired, quic_events.PingAcknowledged, quic_events.ProtocolNegotiated)):
                pass
            else:
                raise AssertionError(f'Unexpected event: {event!r}')
        yield from self.tls_interact()

    def receive_close(self) -> layer.CommandGenerator[None]:
        if False:
            print('Hello World!')
        assert self.quic
        close_event = self.quic._close_event or quic_events.ConnectionTerminated(QuicErrorCode.NO_ERROR, None, 'Connection closed.')
        yield from self.event_to_child(QuicConnectionClosed(self.conn, close_event.error_code, close_event.frame_type, close_event.reason_phrase))

    def send_data(self, data: bytes) -> layer.CommandGenerator[None]:
        if False:
            print('Hello World!')
        assert self.quic
        if data:
            self.quic.send_datagram_frame(data)
        yield from self.tls_interact()

    def send_close(self, command: commands.CloseConnection) -> layer.CommandGenerator[None]:
        if False:
            return 10
        if self.quic:
            if isinstance(command, CloseQuicConnection):
                self.quic.close(command.error_code, command.frame_type, command.reason_phrase)
            else:
                self.quic.close()
            yield from self.tls_interact()
        yield from super().send_close(command)

class ServerQuicLayer(QuicLayer):
    """
    This layer establishes QUIC for a single server connection.
    """
    wait_for_clienthello: bool = False

    def __init__(self, context: context.Context, conn: connection.Server | None=None, time: Callable[[], float] | None=None):
        if False:
            return 10
        super().__init__(context, conn or context.server, time)

    def start_handshake(self) -> layer.CommandGenerator[None]:
        if False:
            print('Hello World!')
        wait_for_clienthello = not self.command_to_reply_to and isinstance(self.child_layer, ClientQuicLayer)
        if wait_for_clienthello:
            self.wait_for_clienthello = True
            self.tunnel_state = tunnel.TunnelState.CLOSED
        else:
            yield from self.start_tls(None)

    def event_to_child(self, event: events.Event) -> layer.CommandGenerator[None]:
        if False:
            i = 10
            return i + 15
        if self.wait_for_clienthello:
            for command in super().event_to_child(event):
                if isinstance(command, commands.OpenConnection) and command.connection == self.conn:
                    self.wait_for_clienthello = False
                else:
                    yield command
        else:
            yield from super().event_to_child(event)

    def on_handshake_error(self, err: str) -> layer.CommandGenerator[None]:
        if False:
            for i in range(10):
                print('nop')
        yield commands.Log(f'Server QUIC handshake failed. {err}', level=WARNING)
        yield from super().on_handshake_error(err)

class ClientQuicLayer(QuicLayer):
    """
    This layer establishes QUIC on a single client connection.
    """
    server_tls_available: bool
    'Indicates whether the parent layer is a ServerQuicLayer.'

    def __init__(self, context: context.Context, time: Callable[[], float] | None=None) -> None:
        if False:
            return 10
        if context.client.tls:
            context.client.alpn = None
            context.client.cipher = None
            context.client.sni = None
            context.client.timestamp_tls_setup = None
            context.client.tls_version = None
            context.client.certificate_list = []
            context.client.mitmcert = None
            context.client.alpn_offers = []
            context.client.cipher_list = []
        super().__init__(context, context.client, time)
        self.server_tls_available = len(self.context.layers) >= 2 and isinstance(self.context.layers[-2], ServerQuicLayer)

    def start_handshake(self) -> layer.CommandGenerator[None]:
        if False:
            return 10
        yield from ()

    def receive_handshake_data(self, data: bytes) -> layer.CommandGenerator[tuple[bool, str | None]]:
        if False:
            i = 10
            return i + 15
        if isinstance(self.context.layers[0], TransparentProxy):
            yield commands.Log(f'Swallowing QUIC handshake because HTTP/3 does not support transparent mode yet.', DEBUG)
            return (False, None)
        if not self.context.options.http3:
            yield commands.Log(f'Swallowing QUIC handshake because HTTP/3 is disabled.', DEBUG)
            return (False, None)
        if self.tls:
            return (yield from super().receive_handshake_data(data))
        buffer = QuicBuffer(data=data)
        try:
            header = pull_quic_header(buffer)
        except TypeError:
            return (False, f'Cannot parse QUIC header: Malformed head ({data.hex()})')
        except ValueError as e:
            return (False, f'Cannot parse QUIC header: {e} ({data.hex()})')
        supported_versions = [version.value for version in QuicProtocolVersion if version is not QuicProtocolVersion.NEGOTIATION]
        if header.version is not None and header.version not in supported_versions:
            yield commands.SendData(self.tunnel_connection, encode_quic_version_negotiation(source_cid=header.destination_cid, destination_cid=header.source_cid, supported_versions=supported_versions))
            return (False, None)
        if len(data) < 1200 or header.packet_type != PACKET_TYPE_INITIAL:
            return (False, f'Invalid handshake received, roaming not supported. ({data.hex()})')
        try:
            client_hello = quic_parse_client_hello(data)
        except ValueError as e:
            return (False, f'Cannot parse ClientHello: {str(e)} ({data.hex()})')
        self.conn.sni = client_hello.sni
        self.conn.alpn_offers = client_hello.alpn_protocols
        tls_clienthello = ClientHelloData(self.context, client_hello)
        yield TlsClienthelloHook(tls_clienthello)
        if tls_clienthello.ignore_connection:
            self.conn = self.tunnel_connection = connection.Client(peername=('ignore-conn', 0), sockname=('ignore-conn', 0), transport_protocol='udp', state=connection.ConnectionState.OPEN)
            parent_layer = self.context.layers[self.context.layers.index(self) - 1]
            if isinstance(parent_layer, ServerQuicLayer):
                parent_layer.conn = parent_layer.tunnel_connection = connection.Server(address=None)
            replacement_layer = UDPLayer(self.context, ignore=True)
            parent_layer.handle_event = replacement_layer.handle_event
            parent_layer._handle_event = replacement_layer._handle_event
            yield from parent_layer.handle_event(events.Start())
            yield from parent_layer.handle_event(events.DataReceived(self.context.client, data))
            return (True, None)
        if tls_clienthello.establish_server_tls_first and (not self.context.server.tls_established):
            err = (yield from self.start_server_tls())
            if err:
                yield commands.Log(f'Unable to establish QUIC connection with server ({err}). Trying to establish QUIC with client anyway. If you plan to redirect requests away from this server, consider setting `connection_strategy` to `lazy` to suppress early connections.')
        yield from self.start_tls(header.destination_cid)
        if not self.conn.connected:
            return (False, 'connection closed early')
        return (yield from super().receive_handshake_data(data))

    def start_server_tls(self) -> layer.CommandGenerator[str | None]:
        if False:
            for i in range(10):
                print('nop')
        if not self.server_tls_available:
            return f'No server QUIC available.'
        err = (yield commands.OpenConnection(self.context.server))
        return err

    def on_handshake_error(self, err: str) -> layer.CommandGenerator[None]:
        if False:
            print('Hello World!')
        yield commands.Log(f'Client QUIC handshake failed. {err}', level=WARNING)
        yield from super().on_handshake_error(err)
        self.event_to_child = self.errored

    def errored(self, event: events.Event) -> layer.CommandGenerator[None]:
        if False:
            i = 10
            return i + 15
        if self.debug is not None:
            yield commands.Log(f'{self.debug}[quic] Swallowing {event} as handshake failed.', DEBUG)