import os
import socket
import weakref
from calibre_extensions.speedup import utf8_decode, websocket_mask as fast_mask
from collections import deque
from hashlib import sha1
from struct import error as struct_error, pack, unpack_from
from threading import Lock
from calibre import as_unicode
from calibre.srv.http_response import HTTPConnection, create_http_handler
from calibre.srv.loop import RDWR, READ, WRITE, Connection, HandleInterrupt, ServerLoop
from calibre.srv.utils import DESIRED_SEND_BUFFER_SIZE
from calibre.utils.speedups import ReadOnlyFileBuffer
from polyglot import http_client
from polyglot.binary import as_base64_unicode
from polyglot.queue import Empty, Queue
HANDSHAKE_STR = 'HTTP/1.1 101 Switching Protocols\r\nUpgrade: WebSocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n'
GUID_STR = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
CONTINUATION = 0
TEXT = 1
BINARY = 2
CLOSE = 8
PING = 9
PONG = 10
CONTROL_CODES = (CLOSE, PING, PONG)
ALL_CODES = CONTROL_CODES + (CONTINUATION, TEXT, BINARY)
CHUNK_SIZE = 16 * 1024
SEND_CHUNK_SIZE = DESIRED_SEND_BUFFER_SIZE - 16
NORMAL_CLOSE = 1000
SHUTTING_DOWN = 1001
PROTOCOL_ERROR = 1002
UNSUPPORTED_DATA = 1003
INCONSISTENT_DATA = 1007
POLICY_VIOLATION = 1008
MESSAGE_TOO_BIG = 1009
UNEXPECTED_ERROR = 1011
RESERVED_CLOSE_CODES = (1004, 1005, 1006)

class ReadFrame:

    def __init__(self):
        if False:
            print('Hello World!')
        self.header_buf = bytearray(14)
        self.rbuf = bytearray(CHUNK_SIZE)
        self.empty = memoryview(b'')
        self.reset()

    def reset(self):
        if False:
            return 10
        self.header_view = memoryview(self.header_buf)[:6]
        self.state = self.read_header

    def __call__(self, conn):
        if False:
            while True:
                i = 10
        return self.state(conn)

    def read_header(self, conn):
        if False:
            return 10
        num_bytes = conn.recv_into(self.header_view)
        if num_bytes == 0:
            return
        read_bytes = 6 - len(self.header_view) + num_bytes
        if read_bytes > 2:
            (b1, b2) = (self.header_buf[0], self.header_buf[1])
            self.fin = bool(b1 & 128)
            if b1 & 112:
                conn.log.error('RSV bits set in frame from client')
                conn.websocket_close(PROTOCOL_ERROR, 'RSV bits set')
                return
            self.opcode = b1 & 15
            self.is_control = self.opcode in CONTROL_CODES
            if self.opcode not in ALL_CODES:
                conn.log.error('Unknown OPCODE from client: %r' % self.opcode)
                conn.websocket_close(PROTOCOL_ERROR, 'Unknown OPCODE: %r' % self.opcode)
                return
            if not self.fin and self.is_control:
                conn.log.error('Fragmented control frame from client')
                conn.websocket_close(PROTOCOL_ERROR, 'Fragmented control frame')
                return
            mask = b2 & 128
            if not mask:
                conn.log.error('Unmasked packet from client')
                conn.websocket_close(PROTOCOL_ERROR, 'Unmasked packet not allowed')
                self.reset()
                return
            self.payload_length = l = b2 & 127
            if self.is_control and l > 125:
                conn.log.error('Too large control frame from client')
                conn.websocket_close(PROTOCOL_ERROR, 'Control frame too large')
                self.reset()
                return
            header_len = 6 + (0 if l < 126 else 2 if l == 126 else 8)
            if header_len <= read_bytes:
                self.process_header(conn)
            else:
                self.header_view = memoryview(self.header_buf)[read_bytes:header_len]
                self.state = self.finish_reading_header
        else:
            self.header_view = self.header_view[num_bytes:]

    def finish_reading_header(self, conn):
        if False:
            i = 10
            return i + 15
        num_bytes = conn.recv_into(self.header_view)
        if num_bytes == 0:
            return
        if num_bytes >= len(self.header_view):
            self.process_header(conn)
        else:
            self.header_view = self.header_view[num_bytes:]

    def process_header(self, conn):
        if False:
            i = 10
            return i + 15
        if self.payload_length < 126:
            self.mask = memoryview(self.header_buf)[2:6]
        elif self.payload_length == 126:
            (self.payload_length,) = unpack_from(b'!H', self.header_buf, 2)
            self.mask = memoryview(self.header_buf)[4:8]
        else:
            (self.payload_length,) = unpack_from(b'!Q', self.header_buf, 2)
            self.mask = memoryview(self.header_buf)[10:14]
        self.frame_starting = True
        self.bytes_received = 0
        if self.payload_length <= CHUNK_SIZE:
            if self.payload_length == 0:
                conn.ws_data_received(self.empty, self.opcode, True, True, self.fin)
                self.reset()
            else:
                self.rview = memoryview(self.rbuf)[:self.payload_length]
                self.state = self.read_packet
        else:
            self.rview = memoryview(self.rbuf)
            self.state = self.read_payload

    def read_packet(self, conn):
        if False:
            i = 10
            return i + 15
        num_bytes = conn.recv_into(self.rview)
        if num_bytes == 0:
            return
        if num_bytes >= len(self.rview):
            data = memoryview(self.rbuf)[:self.payload_length]
            fast_mask(data, self.mask)
            conn.ws_data_received(data, self.opcode, True, True, self.fin)
            self.reset()
        else:
            self.rview = self.rview[num_bytes:]

    def read_payload(self, conn):
        if False:
            while True:
                i = 10
        num_bytes = conn.recv_into(self.rview, min(len(self.rview), self.payload_length - self.bytes_received))
        if num_bytes == 0:
            return
        data = memoryview(self.rbuf)[:num_bytes]
        fast_mask(data, self.mask, self.bytes_received)
        self.bytes_received += num_bytes
        frame_finished = self.bytes_received >= self.payload_length
        conn.ws_data_received(data, self.opcode, self.frame_starting, frame_finished, self.fin)
        self.frame_starting = False
        if frame_finished:
            self.reset()

def create_frame(fin, opcode, payload, mask=None, rsv=0):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(payload, str):
        payload = payload.encode('utf-8')
    l = len(payload)
    header_len = 2 + (0 if l < 126 else 2 if 126 <= l <= 65535 else 8) + (0 if mask is None else 4)
    frame = bytearray(header_len + l)
    if l > 0:
        frame[-l:] = payload
    frame[0] = opcode & 15 | (128 if fin else 0) | rsv & 112
    if l < 126:
        frame[1] = l
    elif 126 <= l <= 65535:
        frame[2:4] = pack(b'!H', l)
        frame[1] = 126
    else:
        frame[2:10] = pack(b'!Q', l)
        frame[1] = 127
    if mask is not None:
        frame[1] |= 128
        frame[header_len - 4:header_len] = mask
        if l > 0:
            fast_mask(memoryview(frame)[-l:], mask)
    return memoryview(frame)

class MessageWriter:

    def __init__(self, buf, mask=None, chunk_size=None):
        if False:
            while True:
                i = 10
        (self.buf, self.data_type, self.mask) = (buf, BINARY, mask)
        if isinstance(buf, str):
            (self.buf, self.data_type) = (ReadOnlyFileBuffer(buf.encode('utf-8')), TEXT)
        elif isinstance(buf, bytes):
            self.buf = ReadOnlyFileBuffer(buf)
        buf = self.buf
        self.chunk_size = chunk_size or SEND_CHUNK_SIZE
        try:
            pos = buf.tell()
            buf.seek(0, os.SEEK_END)
            self.size = buf.tell() - pos
            buf.seek(pos)
        except Exception:
            self.size = None
        self.first_frame_created = self.exhausted = False

    def create_frame(self):
        if False:
            print('Hello World!')
        if self.exhausted:
            return None
        buf = self.buf
        raw = buf.read(self.chunk_size)
        has_more = True if self.size is None else self.size > buf.tell()
        fin = 0 if has_more and raw else 1
        opcode = 0 if self.first_frame_created else self.data_type
        (self.first_frame_created, self.exhausted) = (True, bool(fin))
        return ReadOnlyFileBuffer(create_frame(fin, opcode, raw, self.mask))
conn_id = 0

class UTF8Decoder:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.reset()

    def __call__(self, data):
        if False:
            i = 10
            return i + 15
        (ans, self.state, self.codep) = utf8_decode(data, self.state, self.codep)
        return ans

    def reset(self):
        if False:
            return 10
        self.state = 0
        self.codep = 0

class WebSocketConnection(HTTPConnection):
    in_websocket_mode = False
    websocket_handler = None

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        global conn_id
        HTTPConnection.__init__(self, *args, **kwargs)
        self.sendq = Queue()
        self.control_frames = deque()
        self.cf_lock = Lock()
        self.sending = None
        self.send_buf = None
        self.frag_decoder = UTF8Decoder()
        self.ws_close_received = self.ws_close_sent = False
        conn_id += 1
        self.websocket_connection_id = conn_id
        self.stop_reading = False

    def finalize_headers(self, inheaders):
        if False:
            print('Hello World!')
        upgrade = inheaders.get('Upgrade', '')
        key = inheaders.get('Sec-WebSocket-Key', None)
        conn = {x.strip().lower() for x in inheaders.get('Connection', '').split(',')}
        if key is None or upgrade.lower() != 'websocket' or 'upgrade' not in conn:
            return HTTPConnection.finalize_headers(self, inheaders)
        ver = inheaders.get('Sec-WebSocket-Version', 'Unknown')
        try:
            ver_ok = int(ver) >= 13
        except Exception:
            ver_ok = False
        if not ver_ok:
            return self.simple_response(http_client.BAD_REQUEST, 'Unsupported WebSocket protocol version: %s' % ver)
        if self.method != 'GET':
            return self.simple_response(http_client.BAD_REQUEST, 'Invalid WebSocket method: %s' % self.method)
        response = HANDSHAKE_STR % as_base64_unicode(sha1((key + GUID_STR).encode('utf-8')).digest())
        self.optimize_for_sending_packet()
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.set_state(WRITE, self.upgrade_connection_to_ws, ReadOnlyFileBuffer(response.encode('ascii')), inheaders)

    def upgrade_connection_to_ws(self, buf, inheaders, event):
        if False:
            i = 10
            return i + 15
        if self.write(buf):
            if self.websocket_handler is None:
                self.websocket_handler = DummyHandler()
            (self.read_frame, self.current_recv_opcode) = (ReadFrame(), None)
            self.in_websocket_mode = True
            try:
                self.websocket_handler.handle_websocket_upgrade(self.websocket_connection_id, weakref.ref(self), inheaders)
            except Exception as err:
                self.log.exception('Error in WebSockets upgrade handler:')
                self.websocket_close(UNEXPECTED_ERROR, 'Unexpected error in handler: %r' % as_unicode(err))
            self.handle_event = self.ws_duplex
            self.set_ws_state()
            self.end_send_optimization()

    def set_ws_state(self):
        if False:
            for i in range(10):
                print('nop')
        if self.ws_close_sent or self.ws_close_received:
            if self.ws_close_sent:
                self.ready = False
            else:
                self.wait_for = WRITE
            return
        if self.send_buf is not None or self.sending is not None:
            self.wait_for = RDWR
        else:
            try:
                self.sending = self.sendq.get_nowait()
            except Empty:
                with self.cf_lock:
                    if self.control_frames:
                        self.wait_for = RDWR
                    else:
                        self.wait_for = READ
            else:
                self.wait_for = RDWR
        if self.stop_reading:
            if self.wait_for is READ:
                self.ready = False
            elif self.wait_for is RDWR:
                self.wait_for = WRITE

    def ws_duplex(self, event):
        if False:
            while True:
                i = 10
        if event is READ:
            self.ws_read()
        elif event is WRITE:
            self.ws_write()
        self.set_ws_state()

    def ws_read(self):
        if False:
            while True:
                i = 10
        if not self.stop_reading:
            self.read_frame(self)

    def ws_data_received(self, data, opcode, frame_starting, frame_finished, is_final_frame_of_message):
        if False:
            return 10
        if opcode in CONTROL_CODES:
            return self.ws_control_frame(opcode, data)
        message_starting = self.current_recv_opcode is None
        if message_starting:
            if opcode == CONTINUATION:
                self.log.error('Client sent continuation frame with no message to continue')
                self.websocket_close(PROTOCOL_ERROR, 'Continuation frame without any message to continue')
                return
            self.current_recv_opcode = opcode
        elif frame_starting and opcode != CONTINUATION:
            self.log.error('Client sent continuation frame with non-zero opcode')
            self.websocket_close(PROTOCOL_ERROR, 'Continuation frame with non-zero opcode')
            return
        message_finished = frame_finished and is_final_frame_of_message
        if self.current_recv_opcode == TEXT:
            if message_starting:
                self.frag_decoder.reset()
            empty_data = len(data) == 0
            try:
                data = self.frag_decoder(data)
            except ValueError:
                self.frag_decoder.reset()
                self.log.error('Client sent undecodeable UTF-8')
                return self.websocket_close(INCONSISTENT_DATA, 'Not valid UTF-8')
            if message_finished:
                if not data and (not empty_data) or self.frag_decoder.state:
                    self.frag_decoder.reset()
                    self.log.error('Client sent undecodeable UTF-8')
                    return self.websocket_close(INCONSISTENT_DATA, 'Not valid UTF-8')
        if message_finished:
            self.current_recv_opcode = None
            self.frag_decoder.reset()
        try:
            self.handle_websocket_data(data, message_starting, message_finished)
        except Exception as err:
            self.log.exception('Error in WebSockets data handler:')
            self.websocket_close(UNEXPECTED_ERROR, 'Unexpected error in handler: %r' % as_unicode(err))

    def ws_control_frame(self, opcode, data):
        if False:
            for i in range(10):
                print('nop')
        if opcode in (PING, CLOSE):
            rcode = PONG if opcode == PING else CLOSE
            if opcode == CLOSE:
                self.ws_close_received = True
                self.stop_reading = True
                if data:
                    try:
                        close_code = unpack_from(b'!H', data)[0]
                    except struct_error:
                        data = pack(b'!H', PROTOCOL_ERROR) + b'close frame data must be at least two bytes'
                    else:
                        try:
                            utf8_decode(data[2:])
                        except ValueError:
                            data = pack(b'!H', PROTOCOL_ERROR) + b'close frame data must be valid UTF-8'
                        else:
                            if close_code < 1000 or close_code in RESERVED_CLOSE_CODES or 1011 < close_code < 3000:
                                data = pack(b'!H', PROTOCOL_ERROR) + b'close code reserved'
                else:
                    close_code = NORMAL_CLOSE
                    data = pack(b'!H', close_code)
            f = ReadOnlyFileBuffer(create_frame(1, rcode, data))
            f.is_close_frame = opcode == CLOSE
            with self.cf_lock:
                self.control_frames.append(f)
        elif opcode == PONG:
            try:
                self.websocket_handler.handle_websocket_pong(self.websocket_connection_id, data)
            except Exception:
                self.log.exception('Error in PONG handler:')
        self.set_ws_state()

    def websocket_close(self, code=NORMAL_CLOSE, reason=b''):
        if False:
            i = 10
            return i + 15
        if isinstance(reason, str):
            reason = reason.encode('utf-8')
        self.stop_reading = True
        reason = reason[:123]
        if code is None and (not reason):
            f = ReadOnlyFileBuffer(create_frame(1, CLOSE, b''))
        else:
            f = ReadOnlyFileBuffer(create_frame(1, CLOSE, pack(b'!H', code) + reason))
        f.is_close_frame = True
        with self.cf_lock:
            self.control_frames.append(f)
        self.set_ws_state()

    def ws_write(self):
        if False:
            for i in range(10):
                print('nop')
        if self.ws_close_sent:
            return
        if self.send_buf is not None:
            if self.write(self.send_buf):
                self.end_send_optimization()
                if getattr(self.send_buf, 'is_close_frame', False):
                    self.ws_close_sent = True
                self.send_buf = None
        else:
            with self.cf_lock:
                try:
                    self.send_buf = self.control_frames.popleft()
                except IndexError:
                    if self.sending is not None:
                        self.send_buf = self.sending.create_frame()
                        if self.send_buf is None:
                            self.sending = None
            if self.send_buf is not None:
                self.optimize_for_sending_packet()

    def close(self):
        if False:
            while True:
                i = 10
        if self.in_websocket_mode:
            try:
                self.websocket_handler.handle_websocket_close(self.websocket_connection_id)
            except Exception:
                self.log.exception('Error in WebSocket close handler')
            try:
                if self.send_buf is None and (not self.ws_close_sent):
                    self.websocket_close(SHUTTING_DOWN, 'Shutting down')
                    with self.cf_lock:
                        self.write(self.control_frames.pop())
            except Exception:
                pass
            Connection.close(self)
        else:
            HTTPConnection.close(self)

    def send_websocket_message(self, buf, wakeup=True):
        if False:
            for i in range(10):
                print('nop')
        ' Send a complete message. This class will take care of splitting it\n        into appropriate frames automatically. `buf` must be a file like object. '
        self.sendq.put(MessageWriter(buf))
        self.wait_for = RDWR
        if wakeup:
            self.wakeup()

    def send_websocket_frame(self, data, is_first=True, is_last=True):
        if False:
            print('Hello World!')
        ' Useful for streaming handlers that want to break up messages into\n        frames themselves. Note that these frames will be interleaved with\n        control frames, so they should not be too large. '
        opcode = (TEXT if isinstance(data, str) else BINARY) if is_first else CONTINUATION
        fin = 1 if is_last else 0
        frame = create_frame(fin, opcode, data)
        with self.cf_lock:
            self.control_frames.append(ReadOnlyFileBuffer(frame))

    def send_websocket_ping(self, data=b''):
        if False:
            return 10
        ' Send a PING to the remote client, it should reply with a PONG which\n        will be sent to the handle_websocket_pong callback in your handler. '
        if isinstance(data, str):
            data = data.encode('utf-8')
        frame = create_frame(True, PING, data)
        with self.cf_lock:
            self.control_frames.append(ReadOnlyFileBuffer(frame))

    def handle_websocket_data(self, data, message_starting, message_finished):
        if False:
            return 10
        ' Called when some data is received from the remote client. In\n        general the data may not constitute a complete "message", use the\n        message_starting and message_finished flags to re-assemble it into a\n        complete message in the handler. Note that for binary data, data is a\n        mutable object. If you intend to keep it around after this method\n        returns, create a bytestring from it, using tobytes(). '
        self.websocket_handler.handle_websocket_data(self.websocket_connection_id, data, message_starting, message_finished)

class DummyHandler:

    def handle_websocket_upgrade(self, connection_id, connection_ref, inheaders):
        if False:
            while True:
                i = 10
        conn = connection_ref()
        conn.websocket_close(NORMAL_CLOSE, 'No WebSocket handler available')

    def handle_websocket_data(self, connection_id, data, message_starting, message_finished):
        if False:
            for i in range(10):
                print('nop')
        pass

    def handle_websocket_pong(self, connection_id, data):
        if False:
            i = 10
            return i + 15
        pass

    def handle_websocket_close(self, connection_id):
        if False:
            while True:
                i = 10
        pass

class EchoHandler:

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.ws_connections = {}

    def conn(self, cid):
        if False:
            print('Hello World!')
        ans = self.ws_connections.get(cid)
        if ans is not None:
            ans = ans()
        return ans

    def handle_websocket_upgrade(self, connection_id, connection_ref, inheaders):
        if False:
            while True:
                i = 10
        self.ws_connections[connection_id] = connection_ref

    def handle_websocket_data(self, connection_id, data, message_starting, message_finished):
        if False:
            while True:
                i = 10
        self.conn(connection_id).send_websocket_frame(data, message_starting, message_finished)

    def handle_websocket_pong(self, connection_id, data):
        if False:
            print('Hello World!')
        pass

    def handle_websocket_close(self, connection_id):
        if False:
            print('Hello World!')
        self.ws_connections.pop(connection_id, None)

def run_echo_server():
    if False:
        return 10
    s = ServerLoop(create_http_handler(websocket_handler=EchoHandler()))
    with HandleInterrupt(s.wakeup):
        s.serve_forever()
if __name__ == '__main__':
    run_echo_server()