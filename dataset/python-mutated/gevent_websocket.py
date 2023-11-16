import collections
import errno
import re
import hashlib
import base64
from base64 import b64encode, b64decode
import socket
import struct
import logging
from socket import error as SocketError
import gevent
from gunicorn.workers.base_async import ALREADY_HANDLED
logger = logging.getLogger(__name__)
WS_KEY = b'258EAFA5-E914-47DA-95CA-C5AB0DC85B11'

class WebSocketWSGI(object):

    def __init__(self, handler):
        if False:
            return 10
        self.handler = handler

    def verify_client(self, ws):
        if False:
            print('Hello World!')
        pass

    def _get_key_value(self, key_value):
        if False:
            return 10
        if not key_value:
            return
        key_number = int(re.sub('\\D', '', key_value))
        spaces = re.subn(' ', '', key_value)[1]
        if key_number % spaces != 0:
            return
        part = key_number / spaces
        return part

    def __call__(self, environ, start_response):
        if False:
            print('Hello World!')
        if not (environ.get('HTTP_CONNECTION').find('Upgrade') != -1 and environ['HTTP_UPGRADE'].lower() == 'websocket'):
            start_response('400 Bad Request', [('Connection', 'close')])
            return []
        sock = environ['gunicorn.socket']
        version = environ.get('HTTP_SEC_WEBSOCKET_VERSION')
        ws = WebSocket(sock, environ, version)
        handshake_reply = 'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n'
        key = environ.get('HTTP_SEC_WEBSOCKET_KEY')
        if key:
            ws_key = base64.b64decode(key)
            if len(ws_key) != 16:
                start_response('400 Bad Request', [('Connection', 'close')])
                return []
            protocols = []
            subprotocols = environ.get('HTTP_SEC_WEBSOCKET_PROTOCOL')
            ws_protocols = []
            if subprotocols:
                for s in subprotocols.split(','):
                    s = s.strip()
                    if s in protocols:
                        ws_protocols.append(s)
            if ws_protocols:
                handshake_reply += 'Sec-WebSocket-Protocol: %s\r\n' % ', '.join(ws_protocols)
            exts = []
            extensions = environ.get('HTTP_SEC_WEBSOCKET_EXTENSIONS')
            ws_extensions = []
            if extensions:
                for ext in extensions.split(','):
                    ext = ext.strip()
                    if ext in exts:
                        ws_extensions.append(ext)
            if ws_extensions:
                handshake_reply += 'Sec-WebSocket-Extensions: %s\r\n' % ', '.join(ws_extensions)
            key_hash = hashlib.sha1()
            key_hash.update(key.encode())
            key_hash.update(WS_KEY)
            handshake_reply += 'Sec-WebSocket-Origin: %s\r\nSec-WebSocket-Location: ws://%s%s\r\nSec-WebSocket-Version: %s\r\nSec-WebSocket-Accept: %s\r\n\r\n' % (environ.get('HTTP_ORIGIN'), environ.get('HTTP_HOST'), ws.path, version, base64.b64encode(key_hash.digest()).decode())
        else:
            handshake_reply += 'WebSocket-Origin: %s\r\nWebSocket-Location: ws://%s%s\r\n\r\n' % (environ.get('HTTP_ORIGIN'), environ.get('HTTP_HOST'), ws.path)
        sock.sendall(handshake_reply.encode())
        try:
            self.handler(ws)
        except BrokenPipeError:
            pass
        else:
            raise
        return ALREADY_HANDLED

class WebSocket(object):
    """A websocket object that handles the details of
    serialization/deserialization to the socket.

    The primary way to interact with a :class:`WebSocket` object is to
    call :meth:`send` and :meth:`wait` in order to pass messages back
    and forth with the browser.  Also available are the following
    properties:

    path
        The path value of the request.  This is the same as the WSGI PATH_INFO variable, but more convenient.
    protocol
        The value of the Websocket-Protocol header.
    origin
        The value of the 'Origin' header.
    environ
        The full WSGI environment for this request.

    """

    def __init__(self, sock, environ, version=76):
        if False:
            while True:
                i = 10
        '\n        :param socket: The eventlet socket\n        :type socket: :class:`eventlet.greenio.GreenSocket`\n        :param environ: The wsgi environment\n        :param version: The WebSocket spec version to follow (default is 76)\n        '
        self.socket = sock
        self.origin = environ.get('HTTP_ORIGIN')
        self.protocol = environ.get('HTTP_WEBSOCKET_PROTOCOL')
        self.path = environ.get('PATH_INFO')
        self.environ = environ
        self.version = version
        self.websocket_closed = False
        self._buf = ''
        self._msgs = collections.deque()

    @staticmethod
    def encode_hybi(buf, opcode, base64=False):
        if False:
            while True:
                i = 10
        ' Encode a HyBi style WebSocket frame.\n        Optional opcode:\n            0x0 - continuation\n            0x1 - text frame (base64 encode buf)\n            0x2 - binary frame (use raw buf)\n            0x8 - connection close\n            0x9 - ping\n            0xA - pong\n        '
        if base64:
            buf = b64encode(buf)
        else:
            buf = buf.encode()
        b1 = 128 | opcode & 15
        payload_len = len(buf)
        if payload_len <= 125:
            header = struct.pack('>BB', b1, payload_len)
        elif payload_len > 125 and payload_len < 65536:
            header = struct.pack('>BBH', b1, 126, payload_len)
        elif payload_len >= 65536:
            header = struct.pack('>BBQ', b1, 127, payload_len)
        return (header + buf, len(header), 0)

    @staticmethod
    def decode_hybi(buf, base64=False):
        if False:
            for i in range(10):
                print('nop')
        " Decode HyBi style WebSocket packets.\n        Returns:\n            {'fin'          : 0_or_1,\n             'opcode'       : number,\n             'mask'         : 32_bit_number,\n             'hlen'         : header_bytes_number,\n             'length'       : payload_bytes_number,\n             'payload'      : decoded_buffer,\n             'left'         : bytes_left_number,\n             'close_code'   : number,\n             'close_reason' : string}\n        "
        f = {'fin': 0, 'opcode': 0, 'mask': 0, 'hlen': 2, 'length': 0, 'payload': None, 'left': 0, 'close_code': None, 'close_reason': None}
        blen = len(buf)
        f['left'] = blen
        if blen < f['hlen']:
            return f
        (b1, b2) = struct.unpack_from('>BB', buf)
        f['opcode'] = b1 & 15
        f['fin'] = (b1 & 128) >> 7
        has_mask = (b2 & 128) >> 7
        f['length'] = b2 & 127
        if f['length'] == 126:
            f['hlen'] = 4
            if blen < f['hlen']:
                return f
            (f['length'],) = struct.unpack_from('>xxH', buf)
        elif f['length'] == 127:
            f['hlen'] = 10
            if blen < f['hlen']:
                return f
            (f['length'],) = struct.unpack_from('>xxQ', buf)
        full_len = f['hlen'] + has_mask * 4 + f['length']
        if blen < full_len:
            return f
        f['left'] = blen - full_len
        if has_mask:
            f['mask'] = buf[f['hlen']:f['hlen'] + 4]
            b = c = ''
            if f['length'] >= 4:
                data = struct.unpack('<I', buf[f['hlen']:f['hlen'] + 4])[0]
                of1 = f['hlen'] + 4
                b = ''
                for i in range(0, int(f['length'] / 4)):
                    mask = struct.unpack('<I', buf[of1 + 4 * i:of1 + 4 * (i + 1)])[0]
                    b += struct.pack('I', data ^ mask)
            if f['length'] % 4:
                l = f['length'] % 4
                of1 = f['hlen']
                of2 = full_len - l
                c = ''
                for i in range(0, l):
                    mask = struct.unpack('B', buf[of1 + i])[0]
                    data = struct.unpack('B', buf[of2 + i])[0]
                    c += chr(data ^ mask)
            f['payload'] = b + c
        else:
            print('Unmasked frame: %s' % repr(buf))
            f['payload'] = buf[f['hlen'] + has_mask * 4:full_len]
        if base64 and f['opcode'] in [1, 2]:
            try:
                f['payload'] = b64decode(f['payload'])
            except:
                print('Exception while b64decoding buffer: %s' % repr(buf))
                raise
        if f['opcode'] == 8:
            if f['length'] >= 2:
                f['close_code'] = struct.unpack_from('>H', f['payload'])
            if f['length'] > 3:
                f['close_reason'] = f['payload'][2:]
        return f

    @staticmethod
    def _pack_message(message):
        if False:
            while True:
                i = 10
        'Pack the message inside ``00`` and ``FF``\n\n        As per the dataframing section (5.3) for the websocket spec\n        '
        if isinstance(message, str):
            message = message.encode('utf-8')
        packed = '\x00%sÿ' % message
        return packed

    def _parse_messages(self):
        if False:
            for i in range(10):
                print('nop')
        " Parses for messages in the buffer *buf*.  It is assumed that\n        the buffer contains the start character for a message, but that it\n        may contain only part of the rest of the message.\n\n        Returns an array of messages, and the buffer remainder that\n        didn't contain any full messages."
        msgs = []
        end_idx = 0
        buf = self._buf
        while buf:
            if self.version in ['7', '8', '13']:
                frame = self.decode_hybi(buf, base64=False)
                if frame['payload'] == None:
                    break
                elif frame['opcode'] == 8:
                    self.websocket_closed = True
                    break
                else:
                    msgs.append(frame['payload'])
                    if frame['left']:
                        buf = buf[-frame['left']:]
                    else:
                        buf = ''
            else:
                frame_type = ord(buf[0])
                if frame_type == 0:
                    end_idx = buf.find('ÿ')
                    if end_idx == -1:
                        break
                    msgs.append(buf[1:end_idx].decode('utf-8', 'replace'))
                    buf = buf[end_idx + 1:]
                elif frame_type == 255:
                    assert ord(buf[1]) == 0, 'Unexpected closing handshake: %r' % buf
                    self.websocket_closed = True
                    break
                else:
                    raise ValueError("Don't understand how to parse this type of message: %r" % buf)
        self._buf = buf
        return msgs

    def send(self, message):
        if False:
            print('Hello World!')
        'Send a message to the browser.\n\n        *message* should be convertible to a string; unicode objects should be\n        encodable as utf-8.  Raises socket.error with errno of 32\n        (broken pipe) if the socket has already been closed by the client.'
        if self.version in ['7', '8', '13']:
            (packed, lenhead, lentail) = self.encode_hybi(message, opcode=1, base64=False)
        else:
            packed = self._pack_message(message)
        try:
            self.socket.sendall(packed)
        finally:
            pass

    def wait(self):
        if False:
            while True:
                i = 10
        'Waits for and deserializes messages.\n\n        Returns a single message; the oldest not yet processed. If the client\n        has already closed the connection, returns None.  This is different\n        from normal socket behavior because the empty string is a valid\n        websocket message.'
        while not self._msgs:
            if self.websocket_closed:
                return None
            delta = self.socket.recv(8096)
            if delta == b'':
                return None
            self._buf += delta
            msgs = self._parse_messages()
            self._msgs.extend(msgs)
        return self._msgs.popleft()

    def _send_closing_frame(self, ignore_send_errors=False):
        if False:
            i = 10
            return i + 15
        'Sends the closing frame to the client, if required.'
        if self.version in ['7', '8', '13'] and (not self.websocket_closed):
            msg = ''
            (buf, h, t) = self.encode_hybi(msg, opcode=8, base64=False)
            self.socket.sendall(buf)
            self.websocket_closed = True
        elif self.version == 76 and (not self.websocket_closed):
            try:
                self.socket.sendall(b'\xff\x00')
            except SocketError:
                if not ignore_send_errors:
                    raise
            self.websocket_closed = True

    def close(self):
        if False:
            return 10
        'Forcibly close the websocket; generally it is preferable to\n        return from the handler method.'
        self._send_closing_frame()
        self.socket.shutdown(True)
        self.socket.close()
import os
import random

def handle(ws):
    if False:
        print('Hello World!')
    '  This is the websocket handler function.  Note that we\n    can dispatch based on path in here, too.'
    if ws.path == '/echo':
        while True:
            m = ws.wait()
            if m is None:
                break
            ws.send(m)
    elif ws.path == '/data':
        for i in range(10000):
            ws.send('0 %s %s\n' % (i, random.random()))
            gevent.sleep(0.1)
wsapp = WebSocketWSGI(handle)

def app(environ, start_response):
    if False:
        return 10
    ' This resolves to the web page or the websocket depending on\n    the path.'
    if environ['PATH_INFO'] == '/' or environ['PATH_INFO'] == '':
        data = open(os.path.join(os.path.dirname(__file__), 'websocket.html')).read()
        data = data % environ
        start_response('200 OK', [('Content-Type', 'text/html'), ('Content-Length', str(len(data)))])
        return [data.encode()]
    else:
        return wsapp(environ, start_response)