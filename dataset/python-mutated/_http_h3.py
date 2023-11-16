from collections.abc import Iterable
from dataclasses import dataclass
from aioquic.h3.connection import FrameUnexpected
from aioquic.h3.connection import H3Connection
from aioquic.h3.connection import H3Event
from aioquic.h3.connection import H3Stream
from aioquic.h3.connection import Headers
from aioquic.h3.connection import HeadersState
from aioquic.h3.connection import StreamType
from aioquic.h3.events import HeadersReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import StreamDataReceived
from aioquic.quic.packet import QuicErrorCode
from mitmproxy import connection
from mitmproxy.proxy import commands
from mitmproxy.proxy import layer
from mitmproxy.proxy.layers.quic import CloseQuicConnection
from mitmproxy.proxy.layers.quic import QuicConnectionClosed
from mitmproxy.proxy.layers.quic import QuicStreamDataReceived
from mitmproxy.proxy.layers.quic import QuicStreamEvent
from mitmproxy.proxy.layers.quic import QuicStreamReset
from mitmproxy.proxy.layers.quic import ResetQuicStream
from mitmproxy.proxy.layers.quic import SendQuicStreamData

@dataclass
class TrailersReceived(H3Event):
    """
    The TrailersReceived event is fired whenever trailers are received.
    """
    trailers: Headers
    'The trailers.'
    stream_id: int
    'The ID of the stream the trailers were received for.'
    stream_ended: bool
    'Whether the STREAM frame had the FIN bit set.'
    push_id: int | None = None
    'The Push ID or `None` if this is not a push.'

@dataclass
class StreamReset(H3Event):
    """
    The StreamReset event is fired whenever a stream is reset by the peer.
    """
    stream_id: int
    'The ID of the stream that was reset.'
    error_code: int
    'The error code indicating why the stream was reset.'
    push_id: int | None = None
    'The Push ID or `None` if this is not a push.'

class MockQuic:
    """
    aioquic intermingles QUIC and HTTP/3. This is something we don't want to do because that makes testing much harder.
    Instead, we mock our QUIC connection object here and then take out the wire data to be sent.
    """

    def __init__(self, conn: connection.Connection, is_client: bool) -> None:
        if False:
            print('Hello World!')
        self.conn = conn
        self.pending_commands: list[commands.Command] = []
        self._next_stream_id: list[int] = [0, 1, 2, 3]
        self._is_client = is_client
        self.configuration = QuicConfiguration(is_client=is_client)
        self._quic_logger = None
        self._remote_max_datagram_frame_size = 0

    def close(self, error_code: int=QuicErrorCode.NO_ERROR, frame_type: int | None=None, reason_phrase: str='') -> None:
        if False:
            for i in range(10):
                print('nop')
        self.pending_commands.append(CloseQuicConnection(self.conn, error_code, frame_type, reason_phrase))

    def get_next_available_stream_id(self, is_unidirectional: bool=False) -> int:
        if False:
            i = 10
            return i + 15
        index = int(is_unidirectional) << 1 | int(not self._is_client)
        stream_id = self._next_stream_id[index]
        self._next_stream_id[index] = stream_id + 4
        return stream_id

    def reset_stream(self, stream_id: int, error_code: int) -> None:
        if False:
            return 10
        self.pending_commands.append(ResetQuicStream(self.conn, stream_id, error_code))

    def send_stream_data(self, stream_id: int, data: bytes, end_stream: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.pending_commands.append(SendQuicStreamData(self.conn, stream_id, data, end_stream))

class LayeredH3Connection(H3Connection):
    """
    Creates a H3 connection using a fake QUIC connection, which allows layer separation.
    Also ensures that headers, data and trailers are sent in that order.
    """

    def __init__(self, conn: connection.Connection, is_client: bool, enable_webtransport: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        self._mock = MockQuic(conn, is_client)
        super().__init__(self._mock, enable_webtransport)

    def _after_send(self, stream_id: int, end_stream: bool) -> None:
        if False:
            print('Hello World!')
        if end_stream:
            self._stream[stream_id].headers_send_state = HeadersState.AFTER_TRAILERS

    def _handle_request_or_push_frame(self, frame_type: int, frame_data: bytes | None, stream: H3Stream, stream_ended: bool) -> list[H3Event]:
        if False:
            for i in range(10):
                print('nop')
        events = super()._handle_request_or_push_frame(frame_type, frame_data, stream, stream_ended)
        for (index, event) in enumerate(events):
            if isinstance(event, HeadersReceived) and self._stream[event.stream_id].headers_recv_state == HeadersState.AFTER_TRAILERS:
                events[index] = TrailersReceived(event.headers, event.stream_id, event.stream_ended, event.push_id)
        return events

    def close_connection(self, error_code: int=QuicErrorCode.NO_ERROR, frame_type: int | None=None, reason_phrase: str='') -> None:
        if False:
            i = 10
            return i + 15
        'Closes the underlying QUIC connection and ignores any incoming events.'
        self._is_done = True
        self._quic.close(error_code, frame_type, reason_phrase)

    def end_stream(self, stream_id: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Ends the given stream if not already done so.'
        stream = self._get_or_create_stream(stream_id)
        if stream.headers_send_state != HeadersState.AFTER_TRAILERS:
            super().send_data(stream_id, b'', end_stream=True)
            stream.headers_send_state = HeadersState.AFTER_TRAILERS

    def get_next_available_stream_id(self, is_unidirectional: bool=False):
        if False:
            for i in range(10):
                print('nop')
        'Reserves and returns the next available stream ID.'
        return self._quic.get_next_available_stream_id(is_unidirectional)

    def get_open_stream_ids(self, push_id: int | None) -> Iterable[int]:
        if False:
            i = 10
            return i + 15
        'Iterates over all non-special open streams, optionally for a given push id.'
        return (stream.stream_id for stream in self._stream.values() if stream.push_id == push_id and stream.stream_type == (None if push_id is None else StreamType.PUSH) and (not (stream.headers_recv_state == HeadersState.AFTER_TRAILERS and stream.headers_send_state == HeadersState.AFTER_TRAILERS)))

    def handle_connection_closed(self, event: QuicConnectionClosed) -> None:
        if False:
            i = 10
            return i + 15
        self._is_done = True

    def handle_stream_event(self, event: QuicStreamEvent) -> list[H3Event]:
        if False:
            print('Hello World!')
        if self._is_done:
            return []
        elif isinstance(event, QuicStreamReset):
            stream = self._get_or_create_stream(event.stream_id)
            stream.ended = True
            stream.headers_recv_state = HeadersState.AFTER_TRAILERS
            return [StreamReset(event.stream_id, event.error_code, stream.push_id)]
        elif isinstance(event, QuicStreamDataReceived):
            if self._get_or_create_stream(event.stream_id).ended:
                self.close_connection(error_code=QuicErrorCode.PROTOCOL_VIOLATION, reason_phrase='stream already ended')
                return []
            else:
                return self.handle_event(StreamDataReceived(event.data, event.end_stream, event.stream_id))
        else:
            raise AssertionError(f'Unexpected event: {event!r}')

    def has_sent_headers(self, stream_id: int) -> bool:
        if False:
            i = 10
            return i + 15
        'Indicates whether headers have been sent over the given stream.'
        try:
            return self._stream[stream_id].headers_send_state != HeadersState.INITIAL
        except KeyError:
            return False

    def reset_stream(self, stream_id: int, error_code: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Resets a stream that hasn't been ended locally yet."
        stream = self._get_or_create_stream(stream_id)
        stream.headers_send_state = HeadersState.AFTER_TRAILERS
        self._quic.reset_stream(stream_id, error_code)

    def send_data(self, stream_id: int, data: bytes, end_stream: bool=False) -> None:
        if False:
            return 10
        'Sends data over the given stream.'
        super().send_data(stream_id, data, end_stream)
        self._after_send(stream_id, end_stream)

    def send_datagram(self, flow_id: int, data: bytes) -> None:
        if False:
            return 10
        raise NotImplementedError()

    def send_headers(self, stream_id: int, headers: Headers, end_stream: bool=False) -> None:
        if False:
            return 10
        'Sends headers over the given stream.'
        stream = self._get_or_create_stream(stream_id)
        if stream.headers_send_state != HeadersState.INITIAL:
            raise FrameUnexpected('initial HEADERS frame is not allowed in this state')
        super().send_headers(stream_id, headers, end_stream)
        self._after_send(stream_id, end_stream)

    def send_trailers(self, stream_id: int, trailers: Headers) -> None:
        if False:
            i = 10
            return i + 15
        'Sends trailers over the given stream and ends it.'
        stream = self._get_or_create_stream(stream_id)
        if stream.headers_send_state != HeadersState.AFTER_HEADERS:
            raise FrameUnexpected('trailing HEADERS frame is not allowed in this state')
        super().send_headers(stream_id, trailers, end_stream=True)
        self._after_send(stream_id, end_stream=True)

    def transmit(self) -> layer.CommandGenerator[None]:
        if False:
            return 10
        'Yields all pending commands for the upper QUIC layer.'
        while self._mock.pending_commands:
            yield self._mock.pending_commands.pop(0)
__all__ = ['LayeredH3Connection', 'StreamReset', 'TrailersReceived']