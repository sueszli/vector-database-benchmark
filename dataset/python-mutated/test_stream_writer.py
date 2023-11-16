"""
Tests for StreamWriter
"""
import io
from unittest import TestCase
from samcli.lib.utils.stream_writer import StreamWriter
from unittest.mock import Mock

class TestStreamWriter(TestCase):

    def test_must_write_to_stream(self):
        if False:
            i = 10
            return i + 15
        buffer = b'something'
        stream_mock = Mock()
        writer = StreamWriter(stream_mock)
        writer.write_str(buffer.decode('utf-8'))
        stream_mock.write.assert_called_once_with(buffer.decode('utf-8'))

    def test_must_flush_underlying_stream(self):
        if False:
            return 10
        stream_mock = Mock()
        writer = StreamWriter(stream_mock)
        writer.flush()
        stream_mock.flush.assert_called_once_with()

    def test_auto_flush_must_be_off_by_default(self):
        if False:
            print('Hello World!')
        stream_mock = Mock()
        writer = StreamWriter(stream_mock)
        writer.write_str('something')
        stream_mock.flush.assert_not_called()

    def test_when_auto_flush_on_flush_after_each_write(self):
        if False:
            for i in range(10):
                print('nop')
        stream_mock = Mock()
        flush_mock = Mock()
        stream_mock.flush = flush_mock
        lines = ['first', 'second', 'third']
        writer = StreamWriter(stream_mock, True)
        for line in lines:
            writer.write_str(line)
            flush_mock.assert_called_once_with()
            flush_mock.reset_mock()