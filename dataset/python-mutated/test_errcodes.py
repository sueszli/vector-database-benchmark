import unittest
from .testutils import ConnectingTestCase, slow, reload
from threading import Thread
from psycopg2 import errorcodes

class ErrocodeTests(ConnectingTestCase):

    @slow
    def test_lookup_threadsafe(self):
        if False:
            return 10
        MAX_CYCLES = 2000
        errs = []

        def f(pg_code='40001'):
            if False:
                return 10
            try:
                errorcodes.lookup(pg_code)
            except Exception as e:
                errs.append(e)
        for __ in range(MAX_CYCLES):
            reload(errorcodes)
            (t1, t2) = (Thread(target=f), Thread(target=f))
            (t1.start(), t2.start())
            (t1.join(), t2.join())
            if errs:
                self.fail('raised {} errors in {} cycles (first is {} {})'.format(len(errs), MAX_CYCLES, errs[0].__class__.__name__, errs[0]))

    def test_ambiguous_names(self):
        if False:
            while True:
                i = 10
        self.assertEqual(errorcodes.lookup('2F004'), 'READING_SQL_DATA_NOT_PERMITTED')
        self.assertEqual(errorcodes.lookup('38004'), 'READING_SQL_DATA_NOT_PERMITTED')
        self.assertEqual(errorcodes.READING_SQL_DATA_NOT_PERMITTED, '38004')
        self.assertEqual(errorcodes.READING_SQL_DATA_NOT_PERMITTED_, '2F004')

def test_suite():
    if False:
        while True:
            i = 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main()