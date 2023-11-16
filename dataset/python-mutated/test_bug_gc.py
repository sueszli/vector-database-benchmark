import psycopg2
import psycopg2.extensions
import unittest
import gc
from .testutils import ConnectingTestCase, skip_if_no_uuid

class StolenReferenceTestCase(ConnectingTestCase):

    @skip_if_no_uuid
    def test_stolen_reference_bug(self):
        if False:
            return 10

        def fish(val, cur):
            if False:
                while True:
                    i = 10
            gc.collect()
            return 42
        UUID = psycopg2.extensions.new_type((2950,), 'UUID', fish)
        psycopg2.extensions.register_type(UUID, self.conn)
        curs = self.conn.cursor()
        curs.execute("select 'b5219e01-19ab-4994-b71e-149225dc51e4'::uuid")
        curs.fetchone()

def test_suite():
    if False:
        print('Hello World!')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main()