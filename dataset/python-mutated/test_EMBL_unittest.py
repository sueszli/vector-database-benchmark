"""Tests for EMBL module (using unittest framework)."""
import unittest
import warnings
from os import path
from Bio import SeqIO
from Bio import BiopythonParserWarning

class EMBLTests(unittest.TestCase):

    def test_embl_content_after_co(self):
        if False:
            i = 10
            return i + 15
        'Test a ValueError is thrown by content after a CO line.'

        def parse_content_after_co():
            if False:
                return 10
            rec = SeqIO.read(path.join('EMBL', 'xx_after_co.embl'), 'embl')
        self.assertRaises(ValueError, parse_content_after_co)
        try:
            parse_content_after_co()
        except ValueError as e:
            self.assertEqual(str(e), "Unexpected content after SQ or CO line: 'XX'")
        else:
            self.assertTrue(False, 'Error message without explanation raised by content after CO line')

    def test_embl_0_line(self):
        if False:
            i = 10
            return i + 15
        'Test SQ line with 0 length sequence.'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            rec = SeqIO.read('EMBL/embl_with_0_line.embl', 'embl')
            self.assertEqual(len(w), 0, 'Unexpected parser warnings: ' + '\n'.join((str(warn.message) for warn in w)))
            self.assertEqual(len(rec), 1740)

    def test_embl_no_coords(self):
        if False:
            return 10
        'Test sequence lines without coordinates.'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', BiopythonParserWarning)
            rec = SeqIO.read('EMBL/101ma_no_coords.embl', 'embl')
            self.assertTrue(w, 'Expected parser warning')
            self.assertEqual([str(_.message) for _ in w], ['EMBL sequence line missing coordinates'] * 3)
            self.assertEqual(len(rec), 154)
            self.assertEqual(rec.seq[:10], 'MVLSEGEWQL')
            self.assertEqual(rec.seq[-10:], 'AKYKELGYQG')

    def test_embl_wrong_dr_line(self):
        if False:
            return 10
        'Test files with wrong DR lines.'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', BiopythonParserWarning)
            record = SeqIO.read('EMBL/RepBase23.02.embl', 'embl')
            self.assertTrue(w, 'Expected parser warning')
            self.assertEqual([str(_.message) for _ in w], ['Malformed DR line in EMBL file.'])
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)