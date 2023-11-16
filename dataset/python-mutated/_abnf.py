import array
import os
import struct
import sys
from threading import Lock
from typing import Callable, Union
from ._exceptions import *
from ._utils import validate_utf8
'\n_abnf.py\nwebsocket - WebSocket client library for Python\n\nCopyright 2023 engn33r\n\nLicensed under the Apache License, Version 2.0 (the "License");\nyou may not use this file except in compliance with the License.\nYou may obtain a copy of the License at\n\n    http://www.apache.org/licenses/LICENSE-2.0\n\nUnless required by applicable law or agreed to in writing, software\ndistributed under the License is distributed on an "AS IS" BASIS,\nWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\nSee the License for the specific language governing permissions and\nlimitations under the License.\n'
try:
    from wsaccel.xormask import XorMaskerSimple

    def _mask(_m, _d) -> bytes:
        if False:
            while True:
                i = 10
        return XorMaskerSimple(_m).process(_d)
except ImportError:
    native_byteorder = sys.byteorder

    def _mask(mask_value: array.array, data_value: array.array) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        datalen = len(data_value)
        int_data_value = int.from_bytes(data_value, native_byteorder)
        int_mask_value = int.from_bytes(mask_value * (datalen // 4) + mask_value[:datalen % 4], native_byteorder)
        return (int_data_value ^ int_mask_value).to_bytes(datalen, native_byteorder)
__all__ = ['ABNF', 'continuous_frame', 'frame_buffer', 'STATUS_NORMAL', 'STATUS_GOING_AWAY', 'STATUS_PROTOCOL_ERROR', 'STATUS_UNSUPPORTED_DATA_TYPE', 'STATUS_STATUS_NOT_AVAILABLE', 'STATUS_ABNORMAL_CLOSED', 'STATUS_INVALID_PAYLOAD', 'STATUS_POLICY_VIOLATION', 'STATUS_MESSAGE_TOO_BIG', 'STATUS_INVALID_EXTENSION', 'STATUS_UNEXPECTED_CONDITION', 'STATUS_BAD_GATEWAY', 'STATUS_TLS_HANDSHAKE_ERROR']
STATUS_NORMAL = 1000
STATUS_GOING_AWAY = 1001
STATUS_PROTOCOL_ERROR = 1002
STATUS_UNSUPPORTED_DATA_TYPE = 1003
STATUS_STATUS_NOT_AVAILABLE = 1005
STATUS_ABNORMAL_CLOSED = 1006
STATUS_INVALID_PAYLOAD = 1007
STATUS_POLICY_VIOLATION = 1008
STATUS_MESSAGE_TOO_BIG = 1009
STATUS_INVALID_EXTENSION = 1010
STATUS_UNEXPECTED_CONDITION = 1011
STATUS_SERVICE_RESTART = 1012
STATUS_TRY_AGAIN_LATER = 1013
STATUS_BAD_GATEWAY = 1014
STATUS_TLS_HANDSHAKE_ERROR = 1015
VALID_CLOSE_STATUS = (STATUS_NORMAL, STATUS_GOING_AWAY, STATUS_PROTOCOL_ERROR, STATUS_UNSUPPORTED_DATA_TYPE, STATUS_INVALID_PAYLOAD, STATUS_POLICY_VIOLATION, STATUS_MESSAGE_TOO_BIG, STATUS_INVALID_EXTENSION, STATUS_UNEXPECTED_CONDITION, STATUS_SERVICE_RESTART, STATUS_TRY_AGAIN_LATER, STATUS_BAD_GATEWAY)

class ABNF:
    """
    ABNF frame class.
    See http://tools.ietf.org/html/rfc5234
    and http://tools.ietf.org/html/rfc6455#section-5.2
    """
    OPCODE_CONT = 0
    OPCODE_TEXT = 1
    OPCODE_BINARY = 2
    OPCODE_CLOSE = 8
    OPCODE_PING = 9
    OPCODE_PONG = 10
    OPCODES = (OPCODE_CONT, OPCODE_TEXT, OPCODE_BINARY, OPCODE_CLOSE, OPCODE_PING, OPCODE_PONG)
    OPCODE_MAP = {OPCODE_CONT: 'cont', OPCODE_TEXT: 'text', OPCODE_BINARY: 'binary', OPCODE_CLOSE: 'close', OPCODE_PING: 'ping', OPCODE_PONG: 'pong'}
    LENGTH_7 = 126
    LENGTH_16 = 1 << 16
    LENGTH_63 = 1 << 63

    def __init__(self, fin: int=0, rsv1: int=0, rsv2: int=0, rsv3: int=0, opcode: int=OPCODE_TEXT, mask: int=1, data: Union[str, bytes]='') -> None:
        if False:
            print('Hello World!')
        '\n        Constructor for ABNF. Please check RFC for arguments.\n        '
        self.fin = fin
        self.rsv1 = rsv1
        self.rsv2 = rsv2
        self.rsv3 = rsv3
        self.opcode = opcode
        self.mask = mask
        if data is None:
            data = ''
        self.data = data
        self.get_mask_key = os.urandom

    def validate(self, skip_utf8_validation: bool=False) -> None:
        if False:
            while True:
                i = 10
        '\n        Validate the ABNF frame.\n\n        Parameters\n        ----------\n        skip_utf8_validation: skip utf8 validation.\n        '
        if self.rsv1 or self.rsv2 or self.rsv3:
            raise WebSocketProtocolException('rsv is not implemented, yet')
        if self.opcode not in ABNF.OPCODES:
            raise WebSocketProtocolException('Invalid opcode %r', self.opcode)
        if self.opcode == ABNF.OPCODE_PING and (not self.fin):
            raise WebSocketProtocolException('Invalid ping frame.')
        if self.opcode == ABNF.OPCODE_CLOSE:
            l = len(self.data)
            if not l:
                return
            if l == 1 or l >= 126:
                raise WebSocketProtocolException('Invalid close frame.')
            if l > 2 and (not skip_utf8_validation) and (not validate_utf8(self.data[2:])):
                raise WebSocketProtocolException('Invalid close frame.')
            code = 256 * self.data[0] + self.data[1]
            if not self._is_valid_close_status(code):
                raise WebSocketProtocolException('Invalid close opcode %r', code)

    @staticmethod
    def _is_valid_close_status(code: int) -> bool:
        if False:
            print('Hello World!')
        return code in VALID_CLOSE_STATUS or 3000 <= code < 5000

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return 'fin=' + str(self.fin) + ' opcode=' + str(self.opcode) + ' data=' + str(self.data)

    @staticmethod
    def create_frame(data: Union[bytes, str], opcode: int, fin: int=1) -> 'ABNF':
        if False:
            print('Hello World!')
        '\n        Create frame to send text, binary and other data.\n\n        Parameters\n        ----------\n        data: str\n            data to send. This is string value(byte array).\n            If opcode is OPCODE_TEXT and this value is unicode,\n            data value is converted into unicode string, automatically.\n        opcode: int\n            operation code. please see OPCODE_MAP.\n        fin: int\n            fin flag. if set to 0, create continue fragmentation.\n        '
        if opcode == ABNF.OPCODE_TEXT and isinstance(data, str):
            data = data.encode('utf-8')
        return ABNF(fin, 0, 0, 0, opcode, 1, data)

    def format(self) -> bytes:
        if False:
            return 10
        '\n        Format this object to string(byte array) to send data to server.\n        '
        if any((x not in (0, 1) for x in [self.fin, self.rsv1, self.rsv2, self.rsv3])):
            raise ValueError('not 0 or 1')
        if self.opcode not in ABNF.OPCODES:
            raise ValueError('Invalid OPCODE')
        length = len(self.data)
        if length >= ABNF.LENGTH_63:
            raise ValueError('data is too long')
        frame_header = chr(self.fin << 7 | self.rsv1 << 6 | self.rsv2 << 5 | self.rsv3 << 4 | self.opcode).encode('latin-1')
        if length < ABNF.LENGTH_7:
            frame_header += chr(self.mask << 7 | length).encode('latin-1')
        elif length < ABNF.LENGTH_16:
            frame_header += chr(self.mask << 7 | 126).encode('latin-1')
            frame_header += struct.pack('!H', length)
        else:
            frame_header += chr(self.mask << 7 | 127).encode('latin-1')
            frame_header += struct.pack('!Q', length)
        if not self.mask:
            return frame_header + self.data
        else:
            mask_key = self.get_mask_key(4)
            return frame_header + self._get_masked(mask_key)

    def _get_masked(self, mask_key: Union[str, bytes]) -> bytes:
        if False:
            i = 10
            return i + 15
        s = ABNF.mask(mask_key, self.data)
        if isinstance(mask_key, str):
            mask_key = mask_key.encode('utf-8')
        return mask_key + s

    @staticmethod
    def mask(mask_key: Union[str, bytes], data: Union[str, bytes]) -> bytes:
        if False:
            print('Hello World!')
        '\n        Mask or unmask data. Just do xor for each byte\n\n        Parameters\n        ----------\n        mask_key: bytes or str\n            4 byte mask.\n        data: bytes or str\n            data to mask/unmask.\n        '
        if data is None:
            data = ''
        if isinstance(mask_key, str):
            mask_key = mask_key.encode('latin-1')
        if isinstance(data, str):
            data = data.encode('latin-1')
        return _mask(array.array('B', mask_key), array.array('B', data))

class frame_buffer:
    _HEADER_MASK_INDEX = 5
    _HEADER_LENGTH_INDEX = 6

    def __init__(self, recv_fn: Callable[[int], int], skip_utf8_validation: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.recv = recv_fn
        self.skip_utf8_validation = skip_utf8_validation
        self.recv_buffer = []
        self.clear()
        self.lock = Lock()

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.header = None
        self.length = None
        self.mask = None

    def has_received_header(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.header is None

    def recv_header(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        header = self.recv_strict(2)
        b1 = header[0]
        fin = b1 >> 7 & 1
        rsv1 = b1 >> 6 & 1
        rsv2 = b1 >> 5 & 1
        rsv3 = b1 >> 4 & 1
        opcode = b1 & 15
        b2 = header[1]
        has_mask = b2 >> 7 & 1
        length_bits = b2 & 127
        self.header = (fin, rsv1, rsv2, rsv3, opcode, has_mask, length_bits)

    def has_mask(self) -> Union[bool, int]:
        if False:
            return 10
        if not self.header:
            return False
        return self.header[frame_buffer._HEADER_MASK_INDEX]

    def has_received_length(self) -> bool:
        if False:
            return 10
        return self.length is None

    def recv_length(self) -> None:
        if False:
            print('Hello World!')
        bits = self.header[frame_buffer._HEADER_LENGTH_INDEX]
        length_bits = bits & 127
        if length_bits == 126:
            v = self.recv_strict(2)
            self.length = struct.unpack('!H', v)[0]
        elif length_bits == 127:
            v = self.recv_strict(8)
            self.length = struct.unpack('!Q', v)[0]
        else:
            self.length = length_bits

    def has_received_mask(self) -> bool:
        if False:
            print('Hello World!')
        return self.mask is None

    def recv_mask(self) -> None:
        if False:
            return 10
        self.mask = self.recv_strict(4) if self.has_mask() else ''

    def recv_frame(self) -> ABNF:
        if False:
            while True:
                i = 10
        with self.lock:
            if self.has_received_header():
                self.recv_header()
            (fin, rsv1, rsv2, rsv3, opcode, has_mask, _) = self.header
            if self.has_received_length():
                self.recv_length()
            length = self.length
            if self.has_received_mask():
                self.recv_mask()
            mask = self.mask
            payload = self.recv_strict(length)
            if has_mask:
                payload = ABNF.mask(mask, payload)
            self.clear()
            frame = ABNF(fin, rsv1, rsv2, rsv3, opcode, has_mask, payload)
            frame.validate(self.skip_utf8_validation)
        return frame

    def recv_strict(self, bufsize: int) -> bytes:
        if False:
            while True:
                i = 10
        shortage = bufsize - sum(map(len, self.recv_buffer))
        while shortage > 0:
            bytes_ = self.recv(min(16384, shortage))
            self.recv_buffer.append(bytes_)
            shortage -= len(bytes_)
        unified = b''.join(self.recv_buffer)
        if shortage == 0:
            self.recv_buffer = []
            return unified
        else:
            self.recv_buffer = [unified[bufsize:]]
            return unified[:bufsize]

class continuous_frame:

    def __init__(self, fire_cont_frame: bool, skip_utf8_validation: bool) -> None:
        if False:
            return 10
        self.fire_cont_frame = fire_cont_frame
        self.skip_utf8_validation = skip_utf8_validation
        self.cont_data = None
        self.recving_frames = None

    def validate(self, frame: ABNF) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not self.recving_frames and frame.opcode == ABNF.OPCODE_CONT:
            raise WebSocketProtocolException('Illegal frame')
        if self.recving_frames and frame.opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY):
            raise WebSocketProtocolException('Illegal frame')

    def add(self, frame: ABNF) -> None:
        if False:
            while True:
                i = 10
        if self.cont_data:
            self.cont_data[1] += frame.data
        else:
            if frame.opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY):
                self.recving_frames = frame.opcode
            self.cont_data = [frame.opcode, frame.data]
        if frame.fin:
            self.recving_frames = None

    def is_fire(self, frame: ABNF) -> Union[bool, int]:
        if False:
            print('Hello World!')
        return frame.fin or self.fire_cont_frame

    def extract(self, frame: ABNF) -> list:
        if False:
            return 10
        data = self.cont_data
        self.cont_data = None
        frame.data = data[1]
        if not self.fire_cont_frame and data[0] == ABNF.OPCODE_TEXT and (not self.skip_utf8_validation) and (not validate_utf8(frame.data)):
            raise WebSocketPayloadException('cannot decode: ' + repr(frame.data))
        return [data[0], frame]