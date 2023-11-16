from __future__ import absolute_import
import unittest2
from st2client.utils import strutil

class StrUtilTestCase(unittest2.TestCase):

    def test_unescape(self):
        if False:
            while True:
                i = 10
        in_str = 'Action execution result double escape \\"stuffs\\".\\r\\n'
        expected = 'Action execution result double escape "stuffs".\r\n'
        out_str = strutil.unescape(in_str)
        self.assertEqual(out_str, expected)

    def test_unicode_string(self):
        if False:
            print('Hello World!')
        in_str = '调用CMS接口删除虚拟目录'
        out_str = strutil.unescape(in_str)
        self.assertEqual(out_str, in_str)

    def test_strip_carriage_returns(self):
        if False:
            print('Hello World!')
        in_str = 'Windows editors introduce\r\nlike a noob in 2017.'
        out_str = strutil.strip_carriage_returns(in_str)
        exp_str = 'Windows editors introduce\nlike a noob in 2017.'
        self.assertEqual(out_str, exp_str)