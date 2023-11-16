"""Tests for transports.py."""
import unittest
from unittest import mock
import asyncio
from asyncio import transports

def tearDownModule():
    if False:
        i = 10
        return i + 15
    asyncio.set_event_loop_policy(None)

class TransportTests(unittest.TestCase):

    def test_ctor_extra_is_none(self):
        if False:
            print('Hello World!')
        transport = asyncio.Transport()
        self.assertEqual(transport._extra, {})

    def test_get_extra_info(self):
        if False:
            print('Hello World!')
        transport = asyncio.Transport({'extra': 'info'})
        self.assertEqual('info', transport.get_extra_info('extra'))
        self.assertIsNone(transport.get_extra_info('unknown'))
        default = object()
        self.assertIs(default, transport.get_extra_info('unknown', default))

    def test_writelines(self):
        if False:
            for i in range(10):
                print('nop')
        writer = mock.Mock()

        class MyTransport(asyncio.Transport):

            def write(self, data):
                if False:
                    while True:
                        i = 10
                writer(data)
        transport = MyTransport()
        transport.writelines([b'line1', bytearray(b'line2'), memoryview(b'line3')])
        self.assertEqual(1, writer.call_count)
        writer.assert_called_with(b'line1line2line3')

    def test_not_implemented(self):
        if False:
            for i in range(10):
                print('nop')
        transport = asyncio.Transport()
        self.assertRaises(NotImplementedError, transport.set_write_buffer_limits)
        self.assertRaises(NotImplementedError, transport.get_write_buffer_size)
        self.assertRaises(NotImplementedError, transport.write, 'data')
        self.assertRaises(NotImplementedError, transport.write_eof)
        self.assertRaises(NotImplementedError, transport.can_write_eof)
        self.assertRaises(NotImplementedError, transport.pause_reading)
        self.assertRaises(NotImplementedError, transport.resume_reading)
        self.assertRaises(NotImplementedError, transport.is_reading)
        self.assertRaises(NotImplementedError, transport.close)
        self.assertRaises(NotImplementedError, transport.abort)

    def test_dgram_not_implemented(self):
        if False:
            print('Hello World!')
        transport = asyncio.DatagramTransport()
        self.assertRaises(NotImplementedError, transport.sendto, 'data')
        self.assertRaises(NotImplementedError, transport.abort)

    def test_subprocess_transport_not_implemented(self):
        if False:
            while True:
                i = 10
        transport = asyncio.SubprocessTransport()
        self.assertRaises(NotImplementedError, transport.get_pid)
        self.assertRaises(NotImplementedError, transport.get_returncode)
        self.assertRaises(NotImplementedError, transport.get_pipe_transport, 1)
        self.assertRaises(NotImplementedError, transport.send_signal, 1)
        self.assertRaises(NotImplementedError, transport.terminate)
        self.assertRaises(NotImplementedError, transport.kill)

    def test_flowcontrol_mixin_set_write_limits(self):
        if False:
            while True:
                i = 10

        class MyTransport(transports._FlowControlMixin, transports.Transport):

            def get_write_buffer_size(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 512
        loop = mock.Mock()
        transport = MyTransport(loop=loop)
        transport._protocol = mock.Mock()
        self.assertFalse(transport._protocol_paused)
        with self.assertRaisesRegex(ValueError, 'high.*must be >= low'):
            transport.set_write_buffer_limits(high=0, low=1)
        transport.set_write_buffer_limits(high=1024, low=128)
        self.assertFalse(transport._protocol_paused)
        self.assertEqual(transport.get_write_buffer_limits(), (128, 1024))
        transport.set_write_buffer_limits(high=256, low=128)
        self.assertTrue(transport._protocol_paused)
        self.assertEqual(transport.get_write_buffer_limits(), (128, 256))
if __name__ == '__main__':
    unittest.main()