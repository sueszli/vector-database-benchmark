"""Tests for Clustalw tool."""
from Bio import MissingExternalDependencyError
import sys
import os
import unittest
from Bio import SeqIO
from Bio import AlignIO
from Bio.Align.Applications import ClustalwCommandline
from Bio.Application import ApplicationError
os.environ['LANG'] = 'C'
clustalw_exe = None
if sys.platform == 'win32':
    try:
        prog_files = os.environ['PROGRAMFILES']
    except KeyError:
        prog_files = 'C:\\Program Files'
    likely_dirs = ['ClustalW2', '', 'Clustal', 'Clustalw', 'Clustalw183', 'Clustalw1.83', 'CTCBioApps\\clustalw\\v1.83']
    likely_exes = ['clustalw2.exe', 'clustalw.exe', 'clustalw1.83.exe']
    for folder in likely_dirs:
        if os.path.isdir(os.path.join(prog_files, folder)):
            for filename in likely_exes:
                if os.path.isfile(os.path.join(prog_files, folder, filename)):
                    clustalw_exe = os.path.join(prog_files, folder, filename)
                    break
            if clustalw_exe:
                break
else:
    from subprocess import getoutput
    output = getoutput('clustalw2 --version')
    if 'not found' not in output and 'not recognized' not in output:
        if 'CLUSTAL' in output and 'Multiple Sequence Alignments' in output:
            clustalw_exe = 'clustalw2'
    if not clustalw_exe:
        output = getoutput('clustalw --version')
        if 'not found' not in output and 'not recognized' not in output:
            if 'CLUSTAL' in output and 'Multiple Sequence Alignments' in output:
                clustalw_exe = 'clustalw'
if not clustalw_exe:
    raise MissingExternalDependencyError('Install clustalw or clustalw2 if you want to use it from Biopython.')

class ClustalWTestCase(unittest.TestCase):
    """Class implementing common functions for ClustalW tests."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        'Shared test procedure used by all tests.'
        self.assertEqual(str(eval(repr(cline))), str(cline))
        input_records = SeqIO.to_dict(SeqIO.parse(cline.infile, 'fasta'), lambda rec: rec.id.replace(':', '_'))
        if cline.newtree:
            tree_file = cline.newtree
        else:
            tree_file = os.path.splitext(cline.infile)[0] + '.dnd'
        self.add_file_to_clean(cline.outfile)
        self.add_file_to_clean(tree_file)
        (output, error) = cline()
        self.assertTrue(output.strip().startswith('CLUSTAL'))
        self.assertEqual(error.strip(), '')
        align = AlignIO.read(cline.outfile, 'clustal')
        output_records = SeqIO.to_dict(SeqIO.parse(cline.outfile, 'clustal'))
        self.assertCountEqual(input_records.keys(), output_records.keys())
        for record in align:
            self.assertEqual(record.seq, output_records[record.id].seq)
            self.assertEqual(str(record.seq).replace('-', ''), input_records[record.id].seq)
        self.assertTrue(os.path.isfile(tree_file))

    def add_file_to_clean(self, filename):
        if False:
            for i in range(10):
                print('nop')
        'Add a file for deferred removal by the tearDown routine.'
        self.files_to_clean.add(filename)

class ClustalWTestErrorConditions(ClustalWTestCase):
    """Test general error conditions."""

    def test_empty_file(self):
        if False:
            print('Hello World!')
        'Test a non-existing input file.'
        input_file = 'does_not_exist.fasta'
        self.assertFalse(os.path.isfile(input_file))
        cline = ClustalwCommandline(clustalw_exe, infile=input_file)
        try:
            (stdout, stderr) = cline()
        except ApplicationError as err:
            message = str(err)
            self.assertTrue('Cannot open sequence file' in message or 'Cannot open input file' in message or 'Non-zero return code ' in message, message)
        else:
            self.fail('expected an ApplicationError')

    def test_single_sequence(self):
        if False:
            print('Hello World!')
        'Test an input file containing a single sequence.'
        input_file = 'Fasta/f001'
        self.assertTrue(os.path.isfile(input_file))
        self.assertEqual(len(list(SeqIO.parse(input_file, 'fasta'))), 1)
        cline = ClustalwCommandline(clustalw_exe, infile=input_file)
        try:
            (stdout, stderr) = cline()
            self.assertIn('cannot do multiple alignment', stdout + stderr)
        except ApplicationError as err:
            pass
        if os.path.isfile(input_file + '.aln'):
            self.add_file_to_clean(input_file + '.aln')

    def test_invalid_sequence(self):
        if False:
            i = 10
            return i + 15
        'Test an input file containing an invalid sequence.'
        input_file = 'Medline/pubmed_result1.txt'
        self.assertTrue(os.path.isfile(input_file))
        cline = ClustalwCommandline(clustalw_exe, infile=input_file)
        with self.assertRaises(ApplicationError) as cm:
            (stdout, stderr) = cline()
            self.fail(f'Should have failed, returned:\n{stdout}\n{stderr}')
        err = str(cm.exception)
        self.assertTrue('invalid format' in err or 'not produced' in err or 'No sequences in file' in err or ('Non-zero return code ' in err))

class ClustalWTestNormalConditions(ClustalWTestCase):
    """Tests for normal conditions."""

    def test_properties(self):
        if False:
            return 10
        'Test passing options via properties.'
        cline = ClustalwCommandline(clustalw_exe)
        cline.infile = 'Fasta/f002'
        cline.outfile = 'temp_test.aln'
        cline.align = True
        self.standard_test_procedure(cline)

    def test_simple_fasta(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a simple fasta input file.'
        input_file = 'Fasta/f002'
        output_file = 'temp_test.aln'
        cline = ClustalwCommandline(clustalw_exe, infile=input_file, outfile=output_file)
        self.standard_test_procedure(cline)

    def test_newtree(self):
        if False:
            print('Hello World!')
        'Test newtree files.'
        input_file = 'Registry/seqs.fasta'
        output_file = 'temp_test.aln'
        newtree_file = 'temp_test.dnd'
        cline = ClustalwCommandline(clustalw_exe, infile=input_file, outfile=output_file, newtree=newtree_file, align=True)
        self.standard_test_procedure(cline)
        cline.newtree = 'temp with space.dnd'
        self.standard_test_procedure(cline)

    def test_large_input_file(self):
        if False:
            while True:
                i = 10
        'Test a large input file.'
        input_file = 'temp_cw_prot.fasta'
        records = list(SeqIO.parse('NBRF/Cw_prot.pir', 'pir'))[:40]
        with open(input_file, 'w') as handle:
            SeqIO.write(records, handle, 'fasta')
        del records
        output_file = 'temp_cw_prot.aln'
        cline = ClustalwCommandline(clustalw_exe, infile=input_file, outfile=output_file)
        self.add_file_to_clean(input_file)
        self.standard_test_procedure(cline)

    def test_input_filename_with_space(self):
        if False:
            for i in range(10):
                print('nop')
        'Test an input filename containing a space.'
        input_file = 'Clustalw/temp horses.fasta'
        with open(input_file, 'w') as handle:
            SeqIO.write(SeqIO.parse('Phylip/hennigian.phy', 'phylip'), handle, 'fasta')
        output_file = 'temp with space.aln'
        cline = ClustalwCommandline(clustalw_exe, infile=input_file, outfile=output_file)
        self.add_file_to_clean(input_file)
        self.standard_test_procedure(cline)

    def test_output_filename_with_spaces(self):
        if False:
            for i in range(10):
                print('nop')
        'Test an output filename containing spaces.'
        input_file = 'GFF/multi.fna'
        output_file = 'temp with space.aln'
        cline = ClustalwCommandline(clustalw_exe, infile=input_file, outfile=output_file)
        self.standard_test_procedure(cline)

class ClustalWTestVersionTwoSpecific(ClustalWTestCase):
    """Tests specific to ClustalW2."""

    def test_statistics(self):
        if False:
            print('Hello World!')
        'Test a statistics file.'
        if clustalw_exe == 'clustalw2':
            input_file = 'Fasta/f002'
            output_file = 'temp_test.aln'
            statistics_file = 'temp_stats.txt'
            cline = ClustalwCommandline(clustalw_exe, infile=input_file, outfile=output_file, stats=statistics_file)
            self.add_file_to_clean(statistics_file)
            self.standard_test_procedure(cline)
            self.assertTrue(os.path.isfile(statistics_file))
        else:
            print('Skipping ClustalW2 specific test.')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)