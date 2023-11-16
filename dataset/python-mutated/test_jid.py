"""
Tests for salt.utils.jid
"""
import datetime
import os
import salt.utils.jid
from tests.support.mock import patch
from tests.support.unit import TestCase

class JidTestCase(TestCase):

    def test_jid_to_time(self):
        if False:
            i = 10
            return i + 15
        test_jid = 20131219110700123489
        expected_jid = '2013, Dec 19 11:07:00.123489'
        self.assertEqual(salt.utils.jid.jid_to_time(test_jid), expected_jid)
        incorrect_jid_length = 2012
        self.assertEqual(salt.utils.jid.jid_to_time(incorrect_jid_length), '')

    def test_is_jid(self):
        if False:
            while True:
                i = 10
        self.assertTrue(salt.utils.jid.is_jid('20131219110700123489'))
        self.assertFalse(salt.utils.jid.is_jid(20131219110700123489))
        self.assertFalse(salt.utils.jid.is_jid('2013121911070012348911111'))

    def test_gen_jid(self):
        if False:
            i = 10
            return i + 15
        now = datetime.datetime(2002, 12, 25, 12, 0, 0, 0)
        with patch('salt.utils.jid._utc_now', return_value=now):
            ret = salt.utils.jid.gen_jid({})
            self.assertEqual(ret, '20021225120000000000')
            with patch('salt.utils.jid.LAST_JID_DATETIME', None):
                ret = salt.utils.jid.gen_jid({'unique_jid': True})
                self.assertEqual(ret, '20021225120000000000_{}'.format(os.getpid()))
                ret = salt.utils.jid.gen_jid({'unique_jid': True})
                self.assertEqual(ret, '20021225120000000001_{}'.format(os.getpid()))

    def test_deprecation_58225(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, salt.utils.jid.gen_jid)
        try:
            salt.utils.jid.gen_jid()
        except TypeError as no_opts:
            self.assertEqual(str(no_opts), "gen_jid() missing 1 required positional argument: 'opts'")