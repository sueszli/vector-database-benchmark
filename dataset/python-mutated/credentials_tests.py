"""
Tests for pika.credentials

"""
import unittest
from unittest import mock
from pika import credentials, spec

class ChildPlainCredentials(credentials.PlainCredentials):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(ChildPlainCredentials, self).__init__(*args, **kwargs)
        self.extra = 'e'

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, ChildPlainCredentials):
            return self.extra == other.extra and super(ChildPlainCredentials, self).__eq__(other)
        return NotImplemented

class ChildExternalCredentials(credentials.ExternalCredentials):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(ChildExternalCredentials, self).__init__(*args, **kwargs)
        self.extra = 'e'

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, ChildExternalCredentials):
            return self.extra == other.extra and super(ChildExternalCredentials, self).__eq__(other)
        return NotImplemented

class PlainCredentialsTests(unittest.TestCase):
    CREDENTIALS = ('guest', 'guest')

    def test_eq(self):
        if False:
            while True:
                i = 10
        self.assertEqual(credentials.PlainCredentials('u', 'p'), credentials.PlainCredentials('u', 'p'))
        self.assertEqual(credentials.PlainCredentials('u', 'p', True), credentials.PlainCredentials('u', 'p', True))
        self.assertEqual(credentials.PlainCredentials('u', 'p', False), credentials.PlainCredentials('u', 'p', False))
        self.assertEqual(credentials.PlainCredentials('u', 'p', False), ChildPlainCredentials('u', 'p', False))
        self.assertEqual(ChildPlainCredentials('u', 'p', False), credentials.PlainCredentials('u', 'p', False))

        class Foreign(object):

            def __eq__(self, other):
                if False:
                    print('Hello World!')
                return 'foobar'
        self.assertEqual(credentials.PlainCredentials('u', 'p', False) == Foreign(), 'foobar')
        self.assertEqual(Foreign() == credentials.PlainCredentials('u', 'p', False), 'foobar')

    def test_ne(self):
        if False:
            i = 10
            return i + 15
        self.assertNotEqual(credentials.PlainCredentials('uu', 'p', False), credentials.PlainCredentials('u', 'p', False))
        self.assertNotEqual(credentials.PlainCredentials('u', 'p', False), credentials.PlainCredentials('uu', 'p', False))
        self.assertNotEqual(credentials.PlainCredentials('u', 'pp', False), credentials.PlainCredentials('u', 'p', False))
        self.assertNotEqual(credentials.PlainCredentials('u', 'p', False), credentials.PlainCredentials('u', 'pp', False))
        self.assertNotEqual(credentials.PlainCredentials('u', 'p', True), credentials.PlainCredentials('u', 'p', False))
        self.assertNotEqual(credentials.PlainCredentials('u', 'p', False), credentials.PlainCredentials('u', 'p', True))
        self.assertNotEqual(credentials.PlainCredentials('uu', 'p', False), ChildPlainCredentials('u', 'p', False))
        self.assertNotEqual(ChildPlainCredentials('u', 'pp', False), credentials.PlainCredentials('u', 'p', False))
        self.assertNotEqual(credentials.PlainCredentials('u', 'p', False), dict(username='u', password='p', erase_on_connect=False))
        self.assertNotEqual(dict(username='u', password='p', erase_on_connect=False), credentials.PlainCredentials('u', 'p', False))

        class Foreign(object):

            def __ne__(self, other):
                if False:
                    return 10
                return 'foobar'
        self.assertEqual(credentials.PlainCredentials('u', 'p', False) != Foreign(), 'foobar')
        self.assertEqual(Foreign() != credentials.PlainCredentials('u', 'p', False), 'foobar')

    def test_response_for(self):
        if False:
            i = 10
            return i + 15
        cred = credentials.PlainCredentials(*self.CREDENTIALS)
        start = spec.Connection.Start()
        self.assertEqual(cred.response_for(start), ('PLAIN', b'\x00guest\x00guest'))

    def test_erase_response_for_no_mechanism_match(self):
        if False:
            for i in range(10):
                print('nop')
        cred = credentials.PlainCredentials(*self.CREDENTIALS)
        start = spec.Connection.Start()
        start.mechanisms = 'FOO BAR BAZ'
        self.assertEqual(cred.response_for(start), (None, None))

    def test_erase_credentials_false(self):
        if False:
            return 10
        cred = credentials.PlainCredentials(*self.CREDENTIALS)
        cred.erase_credentials()
        self.assertEqual((cred.username, cred.password), self.CREDENTIALS)

    def test_erase_credentials_true(self):
        if False:
            return 10
        cred = credentials.PlainCredentials(self.CREDENTIALS[0], self.CREDENTIALS[1], True)
        cred.erase_credentials()
        self.assertEqual((cred.username, cred.password), (None, None))

class ExternalCredentialsTest(unittest.TestCase):

    def test_eq(self):
        if False:
            return 10
        cred_1 = credentials.ExternalCredentials()
        cred_2 = credentials.ExternalCredentials()
        cred_3 = ChildExternalCredentials()
        self.assertEqual(cred_1, cred_2)
        self.assertEqual(cred_2, cred_1)
        cred_1.erase_on_connect = True
        cred_2.erase_on_connect = True
        self.assertEqual(cred_1, cred_2)
        self.assertEqual(cred_2, cred_1)
        cred_1.erase_on_connect = False
        cred_2.erase_on_connect = False
        self.assertEqual(cred_1, cred_2)
        self.assertEqual(cred_2, cred_1)
        cred_1.erase_on_connect = False
        cred_3.erase_on_connect = False
        self.assertEqual(cred_1, cred_3)
        self.assertEqual(cred_3, cred_1)

        class Foreign(object):

            def __eq__(self, other):
                if False:
                    return 10
                return 'foobar'
        self.assertEqual(credentials.ExternalCredentials() == Foreign(), 'foobar')
        self.assertEqual(Foreign() == credentials.ExternalCredentials(), 'foobar')

    def test_ne(self):
        if False:
            return 10
        cred_1 = credentials.ExternalCredentials()
        cred_2 = credentials.ExternalCredentials()
        cred_3 = ChildExternalCredentials()
        cred_1.erase_on_connect = False
        cred_2.erase_on_connect = True
        self.assertNotEqual(cred_1, cred_2)
        self.assertNotEqual(cred_2, cred_1)
        cred_1.erase_on_connect = False
        cred_3.erase_on_connect = True
        self.assertNotEqual(cred_1, cred_3)
        self.assertNotEqual(cred_3, cred_1)
        self.assertNotEqual(cred_1, dict(erase_on_connect=False))
        self.assertNotEqual(dict(erase_on_connect=False), cred_1)

        class Foreign(object):

            def __ne__(self, other):
                if False:
                    print('Hello World!')
                return 'foobar'
        self.assertEqual(credentials.ExternalCredentials() != Foreign(), 'foobar')
        self.assertEqual(Foreign() != credentials.ExternalCredentials(), 'foobar')

    def test_response_for(self):
        if False:
            print('Hello World!')
        cred = credentials.ExternalCredentials()
        start = spec.Connection.Start()
        start.mechanisms = 'PLAIN EXTERNAL'
        self.assertEqual(cred.response_for(start), ('EXTERNAL', b''))

    def test_erase_response_for_no_mechanism_match(self):
        if False:
            i = 10
            return i + 15
        cred = credentials.ExternalCredentials()
        start = spec.Connection.Start()
        start.mechanisms = 'FOO BAR BAZ'
        self.assertEqual(cred.response_for(start), (None, None))

    def test_erase_credentials(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('pika.credentials.LOGGER', autospec=True) as logger:
            cred = credentials.ExternalCredentials()
            cred.erase_credentials()
            logger.debug.assert_called_once_with('Not supported by this Credentials type')