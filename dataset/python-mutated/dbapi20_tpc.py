""" Python DB API 2.0 driver Two Phase Commit compliance test suite.

"""
import unittest

class TwoPhaseCommitTests(unittest.TestCase):
    driver = None

    def connect(self):
        if False:
            while True:
                i = 10
        'Make a database connection.'
        raise NotImplementedError
    _last_id = 0
    _global_id_prefix = 'dbapi20_tpc:'

    def make_xid(self, con):
        if False:
            i = 10
            return i + 15
        id = TwoPhaseCommitTests._last_id
        TwoPhaseCommitTests._last_id += 1
        return con.xid(42, f'{self._global_id_prefix}{id}', 'qualifier')

    def test_xid(self):
        if False:
            while True:
                i = 10
        con = self.connect()
        try:
            xid = con.xid(42, 'global', 'bqual')
        except self.driver.NotSupportedError:
            self.fail('Driver does not support transaction IDs.')
        self.assertEquals(xid[0], 42)
        self.assertEquals(xid[1], 'global')
        self.assertEquals(xid[2], 'bqual')
        xid = con.xid(0, '', '')
        self.assertEquals(tuple(xid), (0, '', ''))
        xid = con.xid(2147483647, 'a' * 64, 'b' * 64)
        self.assertEquals(tuple(xid), (2147483647, 'a' * 64, 'b' * 64))

    def test_tpc_begin(self):
        if False:
            i = 10
            return i + 15
        con = self.connect()
        try:
            xid = self.make_xid(con)
            try:
                con.tpc_begin(xid)
            except self.driver.NotSupportedError:
                self.fail('Driver does not support tpc_begin()')
        finally:
            con.close()

    def test_tpc_commit_without_prepare(self):
        if False:
            return 10
        con = self.connect()
        try:
            xid = self.make_xid(con)
            con.tpc_begin(xid)
            cursor = con.cursor()
            cursor.execute('SELECT 1')
            con.tpc_commit()
        finally:
            con.close()

    def test_tpc_rollback_without_prepare(self):
        if False:
            i = 10
            return i + 15
        con = self.connect()
        try:
            xid = self.make_xid(con)
            con.tpc_begin(xid)
            cursor = con.cursor()
            cursor.execute('SELECT 1')
            con.tpc_rollback()
        finally:
            con.close()

    def test_tpc_commit_with_prepare(self):
        if False:
            return 10
        con = self.connect()
        try:
            xid = self.make_xid(con)
            con.tpc_begin(xid)
            cursor = con.cursor()
            cursor.execute('SELECT 1')
            con.tpc_prepare()
            con.tpc_commit()
        finally:
            con.close()

    def test_tpc_rollback_with_prepare(self):
        if False:
            return 10
        con = self.connect()
        try:
            xid = self.make_xid(con)
            con.tpc_begin(xid)
            cursor = con.cursor()
            cursor.execute('SELECT 1')
            con.tpc_prepare()
            con.tpc_rollback()
        finally:
            con.close()

    def test_tpc_begin_in_transaction_fails(self):
        if False:
            print('Hello World!')
        con = self.connect()
        try:
            xid = self.make_xid(con)
            cursor = con.cursor()
            cursor.execute('SELECT 1')
            self.assertRaises(self.driver.ProgrammingError, con.tpc_begin, xid)
        finally:
            con.close()

    def test_tpc_begin_in_tpc_transaction_fails(self):
        if False:
            while True:
                i = 10
        con = self.connect()
        try:
            xid = self.make_xid(con)
            cursor = con.cursor()
            cursor.execute('SELECT 1')
            self.assertRaises(self.driver.ProgrammingError, con.tpc_begin, xid)
        finally:
            con.close()

    def test_commit_in_tpc_fails(self):
        if False:
            i = 10
            return i + 15
        con = self.connect()
        try:
            xid = self.make_xid(con)
            con.tpc_begin(xid)
            self.assertRaises(self.driver.ProgrammingError, con.commit)
        finally:
            con.close()

    def test_rollback_in_tpc_fails(self):
        if False:
            i = 10
            return i + 15
        con = self.connect()
        try:
            xid = self.make_xid(con)
            con.tpc_begin(xid)
            self.assertRaises(self.driver.ProgrammingError, con.rollback)
        finally:
            con.close()