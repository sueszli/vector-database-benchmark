"""Tests for SeqIO Insdc module."""
import unittest
import warnings
from io import StringIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from seq_tests_common import SeqRecordTestBaseClass
from test_SeqIO import SeqIOConverterTestBaseClass

class TestEmbl(unittest.TestCase):

    def test_annotation1(self):
        if False:
            return 10
        'Check parsing of annotation from EMBL files (1).'
        record = SeqIO.read('EMBL/TRBG361.embl', 'embl')
        self.assertEqual(len(record), 1859)
        self.assertEqual(record.annotations['keywords'], ['beta-glucosidase'])
        self.assertEqual(record.annotations['topology'], 'linear')

    def test_annotation2(self):
        if False:
            for i in range(10):
                print('nop')
        'Check parsing of annotation from EMBL files (2).'
        record = SeqIO.read('EMBL/DD231055_edited.embl', 'embl')
        self.assertEqual(len(record), 315)
        self.assertEqual(record.annotations['keywords'], ['JP 2005522996-A/12', 'test-data', 'lot and lots of keywords for this example', 'multi-line keywords'])
        self.assertEqual(record.annotations['topology'], 'linear')

    def test_annotation3(self):
        if False:
            print('Hello World!')
        'Check parsing of annotation from EMBL files (3).'
        record = SeqIO.read('EMBL/AE017046.embl', 'embl')
        self.assertEqual(len(record), 9609)
        self.assertEqual(record.annotations['keywords'], [''])
        self.assertEqual(record.annotations['topology'], 'circular')

    def test_annotation4(self):
        if False:
            i = 10
            return i + 15
        'Check parsing of annotation from EMBL files (4).'
        record = SeqIO.read('EMBL/location_wrap.embl', 'embl')
        self.assertEqual(len(record), 120)
        self.assertNotIn('keywords', record.annotations)
        self.assertNotIn('topology', record.annotations)

    def test_writing_empty_qualifiers(self):
        if False:
            for i in range(10):
                print('nop')
        f = SeqFeature(SimpleLocation(5, 20, strand=+1), type='region', qualifiers={'empty': None, 'zero': 0, 'one': 1, 'text': 'blah'})
        record = SeqRecord(Seq('A' * 100), 'dummy', features=[f])
        record.annotations['molecule_type'] = 'DNA'
        gbk = record.format('gb')
        self.assertIn(' /empty\n', gbk)
        self.assertIn(' /zero=0\n', gbk)
        self.assertIn(' /one=1\n', gbk)
        self.assertIn(' /text="blah"\n', gbk)

class TestEmblRewrite(SeqRecordTestBaseClass):

    def check_rewrite(self, filename):
        if False:
            return 10
        old = SeqIO.read(filename, 'embl')
        old.dbxrefs = []
        old.annotations['accessions'] = old.annotations['accessions'][:1]
        del old.annotations['references']
        buffer = StringIO()
        self.assertEqual(1, SeqIO.write(old, buffer, 'embl'))
        buffer.seek(0)
        new = SeqIO.read(buffer, 'embl')
        self.compare_record(old, new)

    def test_annotation1(self):
        if False:
            while True:
                i = 10
        'Check writing-and-parsing EMBL file (1).'
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.check_rewrite('EMBL/TRBG361.embl')

    def test_annotation2(self):
        if False:
            while True:
                i = 10
        'Check writing-and-parsing EMBL file (2).'
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.check_rewrite('EMBL/DD231055_edited.embl')

    def test_annotation3(self):
        if False:
            return 10
        'Check writing-and-parsing EMBL file (3).'
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.check_rewrite('EMBL/AE017046.embl')

class ConvertTestsInsdc(SeqIOConverterTestBaseClass):

    def test_conversion(self):
        if False:
            print('Hello World!')
        'Test format conversion by SeqIO.write/SeqIO.parse and SeqIO.convert.'
        tests = [('EMBL/U87107.embl', 'embl'), ('EMBL/TRBG361.embl', 'embl'), ('GenBank/NC_005816.gb', 'gb'), ('GenBank/cor6_6.gb', 'genbank')]
        for (filename, fmt) in tests:
            for (in_format, out_format) in self.formats:
                if in_format != fmt:
                    continue
                self.check_conversion(filename, in_format, out_format)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)