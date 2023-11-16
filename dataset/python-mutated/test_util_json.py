from __future__ import absolute_import
import json
import logging
import mock
import unittest2
from st2client.utils import jsutil
LOG = logging.getLogger(__name__)
DOC = {'a01': 1, 'b01': 2, 'c01': {'c11': 3, 'd12': 4, 'c13': {'c21': 5, 'c22': 6}, 'c14': [7, 8, 9]}}
DOC_IP_ADDRESS = {'ips': {'192.168.1.1': {'hostname': 'router.domain.tld'}, '192.168.1.10': {'hostname': 'server.domain.tld'}}}

class TestGetValue(unittest2.TestCase):

    def test_dot_notation(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(jsutil.get_value(DOC, 'a01'), 1)
        self.assertEqual(jsutil.get_value(DOC, 'c01.c11'), 3)
        self.assertEqual(jsutil.get_value(DOC, 'c01.c13.c22'), 6)
        self.assertEqual(jsutil.get_value(DOC, 'c01.c13'), {'c21': 5, 'c22': 6})
        self.assertListEqual(jsutil.get_value(DOC, 'c01.c14'), [7, 8, 9])

    def test_dot_notation_with_val_error(self):
        if False:
            return 10
        self.assertRaises(ValueError, jsutil.get_value, DOC, None)
        self.assertRaises(ValueError, jsutil.get_value, DOC, '')
        self.assertRaises(ValueError, jsutil.get_value, json.dumps(DOC), 'a01')

    def test_dot_notation_with_key_error(self):
        if False:
            while True:
                i = 10
        self.assertIsNone(jsutil.get_value(DOC, 'd01'))
        self.assertIsNone(jsutil.get_value(DOC, 'a01.a11'))
        self.assertIsNone(jsutil.get_value(DOC, 'c01.c11.c21.c31'))
        self.assertIsNone(jsutil.get_value(DOC, 'c01.c14.c31'))

    def test_ip_address(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(jsutil.get_value(DOC_IP_ADDRESS, 'ips."192.168.1.1"'), {'hostname': 'router.domain.tld'})

    def test_chars_nums_dashes_underscores_calls_simple(self):
        if False:
            print('Hello World!')
        for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_':
            with mock.patch('st2client.utils.jsutil._get_value_simple') as mock_simple:
                jsutil.get_value(DOC, char)
                mock_simple.assert_called_with(DOC, char)

    def test_symbols_calls_complex(self):
        if False:
            while True:
                i = 10
        for char in '`~!@#$%^&&*()=+{}[]|\\;:\'"<>,./?':
            with mock.patch('st2client.utils.jsutil._get_value_complex') as mock_complex:
                jsutil.get_value(DOC, char)
                mock_complex.assert_called_with(DOC, char)

    @mock.patch('st2client.utils.jsutil._get_value_simple')
    def test_single_key_calls_simple(self, mock__get_value_simple):
        if False:
            i = 10
            return i + 15
        jsutil.get_value(DOC, 'a01')
        mock__get_value_simple.assert_called_with(DOC, 'a01')

    @mock.patch('st2client.utils.jsutil._get_value_simple')
    def test_dot_notation_calls_simple(self, mock__get_value_simple):
        if False:
            while True:
                i = 10
        jsutil.get_value(DOC, 'c01.c11')
        mock__get_value_simple.assert_called_with(DOC, 'c01.c11')

    @mock.patch('st2client.utils.jsutil._get_value_complex')
    def test_ip_address_calls_complex(self, mock__get_value_complex):
        if False:
            i = 10
            return i + 15
        jsutil.get_value(DOC_IP_ADDRESS, 'ips."192.168.1.1"')
        mock__get_value_complex.assert_called_with(DOC_IP_ADDRESS, 'ips."192.168.1.1"')

    @mock.patch('st2client.utils.jsutil._get_value_complex')
    def test_beginning_dot_calls_complex(self, mock__get_value_complex):
        if False:
            i = 10
            return i + 15
        jsutil.get_value(DOC, '.c01.c11')
        mock__get_value_complex.assert_called_with(DOC, '.c01.c11')

    @mock.patch('st2client.utils.jsutil._get_value_complex')
    def test_ending_dot_calls_complex(self, mock__get_value_complex):
        if False:
            while True:
                i = 10
        jsutil.get_value(DOC, 'c01.c11.')
        mock__get_value_complex.assert_called_with(DOC, 'c01.c11.')

    @mock.patch('st2client.utils.jsutil._get_value_complex')
    def test_double_dot_calls_complex(self, mock__get_value_complex):
        if False:
            return 10
        jsutil.get_value(DOC, 'c01..c11')
        mock__get_value_complex.assert_called_with(DOC, 'c01..c11')

class TestGetKeyValuePairs(unittest2.TestCase):

    def test_select_kvps(self):
        if False:
            print('Hello World!')
        self.assertEqual(jsutil.get_kvps(DOC, ['a01']), {'a01': 1})
        self.assertEqual(jsutil.get_kvps(DOC, ['c01.c11']), {'c01': {'c11': 3}})
        self.assertEqual(jsutil.get_kvps(DOC, ['c01.c13.c22']), {'c01': {'c13': {'c22': 6}}})
        self.assertEqual(jsutil.get_kvps(DOC, ['c01.c13']), {'c01': {'c13': {'c21': 5, 'c22': 6}}})
        self.assertEqual(jsutil.get_kvps(DOC, ['c01.c14']), {'c01': {'c14': [7, 8, 9]}})
        self.assertEqual(jsutil.get_kvps(DOC, ['a01', 'c01.c11', 'c01.c13.c21']), {'a01': 1, 'c01': {'c11': 3, 'c13': {'c21': 5}}})
        self.assertEqual(jsutil.get_kvps(DOC_IP_ADDRESS, ['ips."192.168.1.1"', 'ips."192.168.1.10".hostname']), {'ips': {'"192': {'168': {'1': {'1"': {'hostname': 'router.domain.tld'}, '10"': {'hostname': 'server.domain.tld'}}}}}})

    def test_select_kvps_with_val_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, jsutil.get_kvps, DOC, [None])
        self.assertRaises(ValueError, jsutil.get_kvps, DOC, [''])
        self.assertRaises(ValueError, jsutil.get_kvps, json.dumps(DOC), ['a01'])

    def test_select_kvps_with_key_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(jsutil.get_kvps(DOC, ['d01']), {})
        self.assertEqual(jsutil.get_kvps(DOC, ['a01.a11']), {})
        self.assertEqual(jsutil.get_kvps(DOC, ['c01.c11.c21.c31']), {})
        self.assertEqual(jsutil.get_kvps(DOC, ['c01.c14.c31']), {})
        self.assertEqual(jsutil.get_kvps(DOC, ['a01', 'c01.c11', 'c01.c13.c23']), {'a01': 1, 'c01': {'c11': 3}})