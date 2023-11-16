import asyncio
import codecs
import dataclasses
import unittest
import unittest.mock
import warnings
from websockets.exceptions import PayloadTooBig, ProtocolError
from websockets.frames import OP_BINARY, OP_CLOSE, OP_PING, OP_PONG, OP_TEXT, CloseCode
from websockets.legacy.framing import *
from .utils import AsyncioTestCase

class FramingTests(AsyncioTestCase):

    def decode(self, message, mask=False, max_size=None, extensions=None):
        if False:
            for i in range(10):
                print('nop')
        stream = asyncio.StreamReader(loop=self.loop)
        stream.feed_data(message)
        stream.feed_eof()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            frame = self.loop.run_until_complete(Frame.read(stream.readexactly, mask=mask, max_size=max_size, extensions=extensions))
        self.assertTrue(stream.at_eof())
        return frame

    def encode(self, frame, mask=False, extensions=None):
        if False:
            while True:
                i = 10
        write = unittest.mock.Mock()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            frame.write(write, mask=mask, extensions=extensions)
        self.assertEqual(write.call_count, 1)
        self.assertEqual(len(write.call_args[0]), 1)
        self.assertEqual(len(write.call_args[1]), 0)
        return write.call_args[0][0]

    def round_trip(self, message, expected, mask=False, extensions=None):
        if False:
            for i in range(10):
                print('nop')
        decoded = self.decode(message, mask, extensions=extensions)
        decoded.check()
        self.assertEqual(decoded, expected)
        encoded = self.encode(decoded, mask, extensions=extensions)
        if mask:
            decoded = self.decode(encoded, mask, extensions=extensions)
            self.assertEqual(decoded, expected)
        else:
            self.assertEqual(encoded, message)

    def test_text(self):
        if False:
            print('Hello World!')
        self.round_trip(b'\x81\x04Spam', Frame(True, OP_TEXT, b'Spam'))

    def test_text_masked(self):
        if False:
            i = 10
            return i + 15
        self.round_trip(b'\x81\x84[\xfb\xe1\xa8\x08\x8b\x80\xc5', Frame(True, OP_TEXT, b'Spam'), mask=True)

    def test_binary(self):
        if False:
            print('Hello World!')
        self.round_trip(b'\x82\x04Eggs', Frame(True, OP_BINARY, b'Eggs'))

    def test_binary_masked(self):
        if False:
            while True:
                i = 10
        self.round_trip(b'\x82\x84S\xcd\xe2\x89\x16\xaa\x85\xfa', Frame(True, OP_BINARY, b'Eggs'), mask=True)

    def test_non_ascii_text(self):
        if False:
            for i in range(10):
                print('nop')
        self.round_trip(b'\x81\x05caf\xc3\xa9', Frame(True, OP_TEXT, 'café'.encode('utf-8')))

    def test_non_ascii_text_masked(self):
        if False:
            i = 10
            return i + 15
        self.round_trip(b'\x81\x85d\xbe\xee~\x07\xdf\x88\xbd\xcd', Frame(True, OP_TEXT, 'café'.encode('utf-8')), mask=True)

    def test_close(self):
        if False:
            while True:
                i = 10
        self.round_trip(b'\x88\x00', Frame(True, OP_CLOSE, b''))

    def test_ping(self):
        if False:
            for i in range(10):
                print('nop')
        self.round_trip(b'\x89\x04ping', Frame(True, OP_PING, b'ping'))

    def test_pong(self):
        if False:
            print('Hello World!')
        self.round_trip(b'\x8a\x04pong', Frame(True, OP_PONG, b'pong'))

    def test_long(self):
        if False:
            print('Hello World!')
        self.round_trip(b'\x82~\x00~' + 126 * b'a', Frame(True, OP_BINARY, 126 * b'a'))

    def test_very_long(self):
        if False:
            while True:
                i = 10
        self.round_trip(b'\x82\x7f\x00\x00\x00\x00\x00\x01\x00\x00' + 65536 * b'a', Frame(True, OP_BINARY, 65536 * b'a'))

    def test_payload_too_big(self):
        if False:
            return 10
        with self.assertRaises(PayloadTooBig):
            self.decode(b'\x82~\x04\x01' + 1025 * b'a', max_size=1024)

    def test_bad_reserved_bits(self):
        if False:
            i = 10
            return i + 15
        for encoded in [b'\xc0\x00', b'\xa0\x00', b'\x90\x00']:
            with self.subTest(encoded=encoded):
                with self.assertRaises(ProtocolError):
                    self.decode(encoded)

    def test_good_opcode(self):
        if False:
            while True:
                i = 10
        for opcode in list(range(0, 3)) + list(range(8, 11)):
            encoded = bytes([128 | opcode, 0])
            with self.subTest(encoded=encoded):
                self.decode(encoded)

    def test_bad_opcode(self):
        if False:
            return 10
        for opcode in list(range(3, 8)) + list(range(11, 16)):
            encoded = bytes([128 | opcode, 0])
            with self.subTest(encoded=encoded):
                with self.assertRaises(ProtocolError):
                    self.decode(encoded)

    def test_mask_flag(self):
        if False:
            i = 10
            return i + 15
        self.decode(b'\x80\x80\x00\x00\x00\x00', mask=True)
        with self.assertRaises(ProtocolError):
            self.decode(b'\x80\x80\x00\x00\x00\x00')
        self.decode(b'\x80\x00')
        with self.assertRaises(ProtocolError):
            self.decode(b'\x80\x00', mask=True)

    def test_control_frame_max_length(self):
        if False:
            print('Hello World!')
        self.decode(b'\x88~\x00}' + 125 * b'a')
        with self.assertRaises(ProtocolError):
            self.decode(b'\x88~\x00~' + 126 * b'a')

    def test_fragmented_control_frame(self):
        if False:
            while True:
                i = 10
        self.decode(b'\x88\x00')
        with self.assertRaises(ProtocolError):
            self.decode(b'\x08\x00')

    def test_extensions(self):
        if False:
            for i in range(10):
                print('nop')

        class Rot13:

            @staticmethod
            def encode(frame):
                if False:
                    for i in range(10):
                        print('nop')
                assert frame.opcode == OP_TEXT
                text = frame.data.decode()
                data = codecs.encode(text, 'rot13').encode()
                return dataclasses.replace(frame, data=data)

            @staticmethod
            def decode(frame, *, max_size=None):
                if False:
                    i = 10
                    return i + 15
                return Rot13.encode(frame)
        self.round_trip(b'\x81\x05uryyb', Frame(True, OP_TEXT, b'hello'), extensions=[Rot13()])

class ParseAndSerializeCloseTests(unittest.TestCase):

    def assertCloseData(self, code, reason, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Serializing code / reason yields data. Parsing data yields code / reason.\n\n        '
        serialized = serialize_close(code, reason)
        self.assertEqual(serialized, data)
        parsed = parse_close(data)
        self.assertEqual(parsed, (code, reason))

    def test_parse_close_and_serialize_close(self):
        if False:
            while True:
                i = 10
        self.assertCloseData(CloseCode.NORMAL_CLOSURE, '', b'\x03\xe8')
        self.assertCloseData(CloseCode.NORMAL_CLOSURE, 'OK', b'\x03\xe8OK')

    def test_parse_close_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(parse_close(b''), (CloseCode.NO_STATUS_RCVD, ''))

    def test_parse_close_errors(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ProtocolError):
            parse_close(b'\x03')
        with self.assertRaises(ProtocolError):
            parse_close(b'\x03\xe7')
        with self.assertRaises(UnicodeDecodeError):
            parse_close(b'\x03\xe8\xff\xff')

    def test_serialize_close_errors(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ProtocolError):
            serialize_close(999, '')