"""Test storing biopython objects in a BioSQL relational db."""
import os
import platform
import unittest
import tempfile
import time
from io import StringIO
import warnings
from Bio import BiopythonWarning
from Bio import MissingExternalDependencyError
from Bio.Seq import Seq, MutableSeq
from Bio.SeqFeature import SeqFeature
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from BioSQL import BioSeqDatabase
from BioSQL import BioSeq
from Bio import Entrez
from common_BioSQL import create_database, destroy_database, check_config
import requires_internet
if __name__ == '__main__':
    raise RuntimeError('Call this via test_BioSQL_*online.py not directly')
SYSTEM = platform.system()

def share_config(dbdriver, dbtype, dbhost, dbuser, dbpasswd, testdb):
    if False:
        i = 10
        return i + 15
    'Make sure we can access the DB settings from this file.'
    global DBDRIVER, DBTYPE, DBHOST, DBUSER, DBPASSWD, TESTDB, DBSCHEMA
    global SYSTEM, SQL_FILE
    DBDRIVER = dbdriver
    DBTYPE = dbtype
    DBHOST = dbhost
    DBUSER = dbuser
    DBPASSWD = dbpasswd
    TESTDB = testdb

class TaxonomyTest(unittest.TestCase):
    """Test proper insertion and retrieval of taxonomy data."""

    def setUp(self):
        if False:
            while True:
                i = 10
        global DBDRIVER, DBTYPE, DBHOST, DBUSER, DBPASSWD, TESTDB, DBSCHEMA
        global SYSTEM, SQL_FILE
        Entrez.email = 'biopython@biopython.org'
        TESTDB = create_database()
        db_name = 'biosql-test'
        self.server = BioSeqDatabase.open_database(driver=DBDRIVER, user=DBUSER, passwd=DBPASSWD, host=DBHOST, db=TESTDB)
        try:
            self.server[db_name]
            self.server.remove_database(db_name)
        except KeyError:
            pass
        self.db = self.server.new_database(db_name)
        self.iterator = SeqIO.parse('GenBank/cor6_6.gb', 'gb')

    def tearDown(self):
        if False:
            return 10
        self.server.close()
        destroy_database()
        del self.db
        del self.server

    def test_taxon_left_right_values(self):
        if False:
            print('Hello World!')
        self.db.load(self.iterator, True)
        sql = "SELECT DISTINCT include.ncbi_taxon_id FROM taxon\n                  INNER JOIN taxon AS include ON\n                      (include.left_value BETWEEN taxon.left_value\n                                  AND taxon.right_value)\n                  WHERE taxon.taxon_id IN\n                      (SELECT taxon_id FROM taxon_name\n                                  WHERE name = 'Brassicales')\n                      AND include.right_value - include.left_value = 1"
        rows = self.db.adaptor.execute_and_fetchall(sql)
        self.assertEqual(4, len(rows))
        values = [row[0] for row in rows]
        self.assertCountEqual([3704, 3711, 3708, 3702], values)

    def test_load_database_with_tax_lookup(self):
        if False:
            print('Hello World!')
        'Load SeqRecord objects and fetch the taxonomy information from NCBI.'
        handle = Entrez.efetch(db='taxonomy', id=3702, retmode='XML')
        taxon_record = Entrez.read(handle)
        entrez_tax = []
        for t in taxon_record[0]['LineageEx']:
            entrez_tax.append(t['ScientificName'])
        entrez_tax.append(taxon_record[0]['ScientificName'])
        self.db.load(self.iterator, True)
        items = list(self.db.values())
        self.assertEqual(len(items), 6)
        self.assertEqual(len(self.db), 6)
        test_record = self.db.lookup(accession='X55053')
        self.assertEqual(test_record.annotations['ncbi_taxid'], 3702)
        self.assertEqual(test_record.annotations['taxonomy'], entrez_tax)