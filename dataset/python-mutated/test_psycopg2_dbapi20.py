from . import dbapi20
from . import dbapi20_tpc
from .testutils import skip_if_tpc_disabled
import unittest
import psycopg2
from .testconfig import dsn

class Psycopg2Tests(dbapi20.DatabaseAPI20Test):
    driver = psycopg2
    connect_args = ()
    connect_kw_args = {'dsn': dsn}
    lower_func = 'lower'

    def test_callproc(self):
        if False:
            return 10
        con = self._connect()
        try:
            cur = con.cursor()
            if self.lower_func and hasattr(cur, 'callproc'):
                cur.callproc(self.lower_func, ('FOO',))
                r = cur.fetchall()
                self.assertEqual(len(r), 1, 'callproc produced no result set')
                self.assertEqual(len(r[0]), 1, 'callproc produced invalid result set')
                self.assertEqual(r[0][0], 'foo', 'callproc produced invalid results')
        finally:
            con.close()

    def test_setoutputsize(self):
        if False:
            print('Hello World!')
        pass

    def test_nextset(self):
        if False:
            for i in range(10):
                print('nop')
        pass

@skip_if_tpc_disabled
class Psycopg2TPCTests(dbapi20_tpc.TwoPhaseCommitTests, unittest.TestCase):
    driver = psycopg2

    def connect(self):
        if False:
            return 10
        return psycopg2.connect(dsn=dsn)

def test_suite():
    if False:
        while True:
            i = 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main()