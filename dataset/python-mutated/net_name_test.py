from unittest.mock import patch
from caffe2.python.test_util import TestCase

class NetNameTest(TestCase):

    @patch('caffe2.python.core.Net.current_prefix', return_value='prefix')
    def test_net_name(self, _current_prefix):
        if False:
            i = 10
            return i + 15
        from caffe2.python.core import Net
        self.assertEqual(Net._get_next_net_name('test'), 'prefix/test')
        self.assertEqual(Net._get_next_net_name('test'), 'prefix/test_1')
        self.assertEqual(Net._get_next_net_name('test_1_2'), 'prefix/test_1_2')
        self.assertEqual(Net._get_next_net_name('test_1'), 'prefix/test_1_1')
        self.assertEqual(Net._get_next_net_name('test_1'), 'prefix/test_1_3')