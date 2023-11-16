from unittest import mock
import torch.distributed.elastic.utils.store as store_util
from torch.distributed.elastic.utils.logging import get_logger
from torch.testing._internal.common_utils import run_tests, TestCase

class StoreUtilTest(TestCase):

    def test_get_all_rank_0(self):
        if False:
            return 10
        store = mock.MagicMock()
        world_size = 3
        store_util.get_all(store, 0, 'test/store', world_size)
        actual_set_call_args = [call_args[0][0] for call_args in store.set.call_args_list]
        self.assertListEqual(['test/store0.FIN'], actual_set_call_args)
        actual_get_call_args = [call_args[0] for call_args in store.get.call_args_list]
        expected_get_call_args = [('test/store0',), ('test/store1',), ('test/store2',), ('test/store0.FIN',), ('test/store1.FIN',), ('test/store2.FIN',)]
        self.assertListEqual(expected_get_call_args, actual_get_call_args)

    def test_get_all_rank_n(self):
        if False:
            return 10
        store = mock.MagicMock()
        world_size = 3
        store_util.get_all(store, 1, 'test/store', world_size)
        actual_set_call_args = [call_args[0][0] for call_args in store.set.call_args_list]
        self.assertListEqual(['test/store1.FIN'], actual_set_call_args)
        actual_get_call_args = [call_args[0] for call_args in store.get.call_args_list]
        expected_get_call_args = [('test/store0',), ('test/store1',), ('test/store2',)]
        self.assertListEqual(expected_get_call_args, actual_get_call_args)

    def test_synchronize(self):
        if False:
            print('Hello World!')
        store_mock = mock.MagicMock()
        data = b'data0'
        store_util.synchronize(store_mock, data, 0, 3, key_prefix='torchelastic/test')
        actual_set_call_args = store_mock.set.call_args_list
        actual_set_call_args = [call_args[0] for call_args in actual_set_call_args]
        expected_set_call_args = [('torchelastic/test0', b'data0'), ('torchelastic/test0.FIN', b'FIN')]
        self.assertListEqual(expected_set_call_args, actual_set_call_args)
        expected_get_call_args = [('torchelastic/test0',), ('torchelastic/test1',), ('torchelastic/test2',), ('torchelastic/test0.FIN',), ('torchelastic/test1.FIN',), ('torchelastic/test2.FIN',)]
        actual_get_call_args = store_mock.get.call_args_list
        actual_get_call_args = [call_args[0] for call_args in actual_get_call_args]
        self.assertListEqual(expected_get_call_args, actual_get_call_args)

class UtilTest(TestCase):

    def test_get_logger_different(self):
        if False:
            print('Hello World!')
        logger1 = get_logger('name1')
        logger2 = get_logger('name2')
        self.assertNotEqual(logger1.name, logger2.name)

    def test_get_logger(self):
        if False:
            while True:
                i = 10
        logger1 = get_logger()
        self.assertEqual(__name__, logger1.name)

    def test_get_logger_none(self):
        if False:
            print('Hello World!')
        logger1 = get_logger(None)
        self.assertEqual(__name__, logger1.name)

    def test_get_logger_custom_name(self):
        if False:
            i = 10
            return i + 15
        logger1 = get_logger('test.module')
        self.assertEqual('test.module', logger1.name)
if __name__ == '__main__':
    run_tests()