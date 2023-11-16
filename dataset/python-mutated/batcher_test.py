import unittest
from unittest.mock import MagicMock
from destination_langchain.batcher import Batcher

class BatcherTestCase(unittest.TestCase):

    def test_add_single_item(self):
        if False:
            return 10
        batch_size = 3
        flush_handler_mock = MagicMock()
        batcher = Batcher(batch_size, flush_handler_mock)
        batcher.add(1)
        self.assertFalse(flush_handler_mock.called)

    def test_add_flushes_batch(self):
        if False:
            print('Hello World!')
        batch_size = 3
        flush_handler_mock = MagicMock()
        batcher = Batcher(batch_size, flush_handler_mock)
        batcher.add(1)
        batcher.add(2)
        batcher.add(3)
        flush_handler_mock.assert_called_once_with([1, 2, 3])

    def test_flush_empty_buffer(self):
        if False:
            while True:
                i = 10
        batch_size = 3
        flush_handler_mock = MagicMock()
        batcher = Batcher(batch_size, flush_handler_mock)
        batcher.flush()
        self.assertFalse(flush_handler_mock.called)

    def test_flush_non_empty_buffer(self):
        if False:
            i = 10
            return i + 15
        batch_size = 3
        flush_handler_mock = MagicMock()
        batcher = Batcher(batch_size, flush_handler_mock)
        batcher.add(1)
        batcher.add(2)
        batcher.flush()
        flush_handler_mock.assert_called_once_with([1, 2])
        self.assertEqual(len(batcher.buffer), 0)

    def test_flush_if_necessary_flushes_batch(self):
        if False:
            return 10
        batch_size = 3
        flush_handler_mock = MagicMock()
        batcher = Batcher(batch_size, flush_handler_mock)
        batcher.add(1)
        batcher.add(2)
        batcher.add(3)
        batcher.add(4)
        batcher.add(5)
        flush_handler_mock.assert_called_once_with([1, 2, 3])
        self.assertEqual(len(batcher.buffer), 2)

    def test_flush_if_necessary_does_not_flush_incomplete_batch(self):
        if False:
            print('Hello World!')
        batch_size = 3
        flush_handler_mock = MagicMock()
        batcher = Batcher(batch_size, flush_handler_mock)
        batcher.add(1)
        batcher.add(2)
        self.assertFalse(flush_handler_mock.called)
        self.assertEqual(len(batcher.buffer), 2)