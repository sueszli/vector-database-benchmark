import unittest
import pymysql
_mysql = pymysql
from pymysql.constants import FIELD_TYPE
from pymysql.tests import base

class TestDBAPISet(unittest.TestCase):

    def test_set_equality(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(pymysql.STRING == pymysql.STRING)

    def test_set_inequality(self):
        if False:
            print('Hello World!')
        self.assertTrue(pymysql.STRING != pymysql.NUMBER)

    def test_set_equality_membership(self):
        if False:
            print('Hello World!')
        self.assertTrue(FIELD_TYPE.VAR_STRING == pymysql.STRING)

    def test_set_inequality_membership(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(FIELD_TYPE.DATE != pymysql.STRING)

class CoreModule(unittest.TestCase):
    """Core _mysql module features."""

    def test_NULL(self):
        if False:
            while True:
                i = 10
        'Should have a NULL constant.'
        self.assertEqual(_mysql.NULL, 'NULL')

    def test_version(self):
        if False:
            while True:
                i = 10
        'Version information sanity.'
        self.assertTrue(isinstance(_mysql.__version__, str))
        self.assertTrue(isinstance(_mysql.version_info, tuple))
        self.assertEqual(len(_mysql.version_info), 5)

    def test_client_info(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(isinstance(_mysql.get_client_info(), str))

    def test_thread_safe(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(isinstance(_mysql.thread_safe(), int))

class CoreAPI(unittest.TestCase):
    """Test _mysql interaction internals."""

    def setUp(self):
        if False:
            return 10
        kwargs = base.PyMySQLTestCase.databases[0].copy()
        kwargs['read_default_file'] = '~/.my.cnf'
        self.conn = _mysql.connect(**kwargs)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.conn.close()

    def test_thread_id(self):
        if False:
            while True:
                i = 10
        tid = self.conn.thread_id()
        self.assertTrue(isinstance(tid, int), "thread_id didn't return an integral value.")
        self.assertRaises(TypeError, self.conn.thread_id, ('evil',), "thread_id shouldn't accept arguments.")

    def test_affected_rows(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.conn.affected_rows(), 0, 'Should return 0 before we do anything.')

    def test_charset_name(self):
        if False:
            return 10
        self.assertTrue(isinstance(self.conn.character_set_name(), str), 'Should return a string.')

    def test_host_info(self):
        if False:
            print('Hello World!')
        assert isinstance(self.conn.get_host_info(), str), 'should return a string'

    def test_proto_info(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(isinstance(self.conn.get_proto_info(), int), 'Should return an int.')

    def test_server_info(self):
        if False:
            while True:
                i = 10
        self.assertTrue(isinstance(self.conn.get_server_info(), str), 'Should return an str.')
if __name__ == '__main__':
    unittest.main()