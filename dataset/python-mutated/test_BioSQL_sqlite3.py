"""Run BioSQL tests using SQLite."""
import os
import unittest
from Bio import SeqIO
from BioSQL import BioSeqDatabase
from seq_tests_common import SeqRecordTestBaseClass
from common_BioSQL import *
from common_BioSQL import check_config, temp_db_filename
DBDRIVER = 'sqlite3'
DBTYPE = 'sqlite'
DBHOST = None
DBUSER = 'root'
DBPASSWD = None
TESTDB = temp_db_filename()
check_config(DBDRIVER, DBTYPE, DBHOST, DBUSER, DBPASSWD, TESTDB)
if False:
    assert not os.path.isfile('BioSQL/cor6_6.db')
    server = BioSeqDatabase.open_database(driver=DBDRIVER, db='BioSQL/cor6_6.db')
    DBSCHEMA = 'biosqldb-' + DBTYPE + '.sql'
    SQL_FILE = os.path.join(os.getcwd(), 'BioSQL', DBSCHEMA)
    assert os.path.isfile(SQL_FILE), SQL_FILE
    server.load_database_sql(SQL_FILE)
    server.commit()
    db = server.new_database('OLD')
    count = db.load(SeqIO.parse('GenBank/cor6_6.gb', 'gb'))
    assert count == 6
    server.commit()
    assert len(db) == 6
    server.close()

class BackwardsCompatibilityTest(SeqRecordTestBaseClass):

    def test_backwards_compatibility(self):
        if False:
            i = 10
            return i + 15
        'Check can re-use an old BioSQL SQLite3 database.'
        original_records = []
        for record in SeqIO.parse('GenBank/cor6_6.gb', 'gb'):
            if record.annotations['molecule_type'] == 'mRNA':
                record.annotations['molecule_type'] = 'DNA'
            original_records.append(record)
        server = BioSeqDatabase.open_database(driver=DBDRIVER, db='BioSQL/cor6_6.db')
        db = server['OLD']
        self.assertEqual(len(db), len(original_records))
        biosql_records = [db.lookup(name=rec.name) for rec in original_records]
        self.compare_records(original_records, biosql_records)
        server.close()
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)