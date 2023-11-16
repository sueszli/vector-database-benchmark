"""Tests for MSAProbs tool."""
import os
import sys
import unittest
from Bio import MissingExternalDependencyError
from Bio import SeqIO
from Bio.Align.Applications import MSAProbsCommandline
from Bio.Application import ApplicationError
from subprocess import getoutput
os.environ['LANG'] = 'C'
msaprobs_exe = None
try:
    output = getoutput('msaprobs -version')
    if output.startswith('MSAPROBS version'):
        msaprobs_exe = 'msaprobs'
except FileNotFoundError:
    pass
if not msaprobs_exe:
    raise MissingExternalDependencyError('Install msaprobs if you want to use MSAProbs from Biopython.')

class MSAProbsTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.files_to_clean = set()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        for filename in self.files_to_clean:
            if os.path.isfile(filename):
                os.remove(filename)

    def standard_test_procedure(self, cline):
        if False:
            while True:
                i = 10
        'Shared testing procedure used by all tests.'
        self.add_file_to_clean(cline.outfile)
        input_records = SeqIO.to_dict(SeqIO.parse(cline.infile, 'fasta'))
        self.assertEqual(str(eval(repr(cline))), str(cline))
        (output, error) = cline()

    def add_file_to_clean(self, filename):
        if False:
            print('Hello World!')
        'Add a file for deferred removal by the tearDown routine.'
        self.files_to_clean.add(filename)

class MSAProbsTestErrorConditions(MSAProbsTestCase):

    def test_empty_file(self):
        if False:
            while True:
                i = 10
        'Test an empty file.'
        input_file = 'does_not_exist.fasta'
        self.assertFalse(os.path.isfile(input_file))
        cline = MSAProbsCommandline(msaprobs_exe, infile=input_file)
        try:
            (stdout, stderr) = cline()
        except ApplicationError as err:
            self.assertTrue('Cannot open sequence file' in str(err) or 'Cannot open input file' in str(err) or 'Non-zero return code ' in str(err), str(err))
        else:
            self.fail(f'Should have failed, returned:\n{stdout}\n{stderr}')

    def test_single_sequence(self):
        if False:
            return 10
        'Test an input file containing a single sequence.'
        input_file = 'Fasta/f001'
        self.assertTrue(os.path.isfile(input_file))
        self.assertEqual(len(list(SeqIO.parse(input_file, 'fasta'))), 1)
        cline = MSAProbsCommandline(msaprobs_exe, infile=input_file)
        try:
            (stdout, stderr) = cline()
        except ApplicationError as err:
            if sys.platform == 'win32':
                expected = 3221225477
            elif sys.platform == 'darwin':
                expected = -11
            else:
                expected = 139
            self.assertEqual(expected, err.returncode)
        else:
            self.fail(f'Should have failed, returned:\n{stdout}\n{stderr}')

    def test_invalid_format(self):
        if False:
            print('Hello World!')
        'Test an input file in an invalid format.'
        input_file = 'Medline/pubmed_result1.txt'
        self.assertTrue(os.path.isfile(input_file))
        cline = MSAProbsCommandline(msaprobs_exe, infile=input_file)
        try:
            (stdout, stderr) = cline()
        except ApplicationError as err:
            self.assertEqual(err.returncode, 1)
        else:
            self.fail(f'Should have failed, returned:\n{stdout}\n{stderr}')

class MSAProbsTestNormalConditions(MSAProbsTestCase):

    def test_simple_fasta(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a simple fasta file.'
        input_file = 'Registry/seqs.fasta'
        output_file = 'temp_test.aln'
        cline = MSAProbsCommandline(msaprobs_exe, infile=input_file, outfile=output_file, clustalw=True)
        self.standard_test_procedure(cline)

    def test_properties(self):
        if False:
            print('Hello World!')
        'Test setting options via properties.'
        input_file = 'Registry/seqs.fasta'
        output_file = 'temp_test.aln'
        cline = MSAProbsCommandline(msaprobs_exe)
        cline.infile = input_file
        cline.outfile = output_file
        cline.clustalw = True
        self.standard_test_procedure(cline)

    def test_input_filename_with_space(self):
        if False:
            i = 10
            return i + 15
        'Test an input filename containing a space.'
        input_file = 'Clustalw/temp horses.fasta'
        with open(input_file, 'w') as handle:
            SeqIO.write(SeqIO.parse('Phylip/hennigian.phy', 'phylip'), handle, 'fasta')
        output_file = 'temp_test.aln'
        cline = MSAProbsCommandline(msaprobs_exe, infile=input_file, outfile=output_file, clustalw=True)
        self.add_file_to_clean(input_file)
        self.standard_test_procedure(cline)

    def test_output_filename_with_spaces(self):
        if False:
            for i in range(10):
                print('nop')
        'Test an output filename containing spaces.'
        input_file = 'Registry/seqs.fasta'
        output_file = 'temp with spaces.aln'
        cline = MSAProbsCommandline(msaprobs_exe, infile=input_file, outfile=output_file, clustalw=True)
        self.standard_test_procedure(cline)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)