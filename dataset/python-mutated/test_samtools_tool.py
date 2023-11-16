"""Tests for samtools tool."""
from Bio import MissingExternalDependencyError
import sys
import os
import unittest
from Bio.Application import ApplicationError
from Bio.Sequencing.Applications import SamtoolsViewCommandline
from Bio.Sequencing.Applications import SamtoolsCalmdCommandline
from Bio.Sequencing.Applications import SamtoolsCatCommandline
from Bio.Sequencing.Applications import SamtoolsFaidxCommandline
from Bio.Sequencing.Applications import SamtoolsIdxstatsCommandline
from Bio.Sequencing.Applications import SamtoolsIndexCommandline
from Bio.Sequencing.Applications import SamtoolsMergeCommandline
from Bio.Sequencing.Applications import SamtoolsMpileupCommandline
from Bio.Sequencing.Applications import SamtoolsVersion1xSortCommandline
from Bio.Sequencing.Applications import SamtoolsSortCommandline
SamtoolsVersion0xSortCommandline = SamtoolsSortCommandline
os.environ['LANG'] = 'C'
samtools_exe = None
if sys.platform == 'win32':
    try:
        prog_files = os.environ['PROGRAMFILES']
    except KeyError:
        prog_files = 'C:\\Program Files'
    likely_dirs = ['samtools', '']
    likely_exes = ['samtools.exe']
    for folder in likely_dirs:
        if os.path.isdir(os.path.join(prog_files, folder)):
            for filename in likely_exes:
                if os.path.isfile(os.path.join(prog_files, folder, filename)):
                    samtools_exe = os.path.join(prog_files, folder, filename)
                    break
            if samtools_exe:
                break
else:
    from subprocess import getoutput
    output = getoutput('samtools')
    if 'not found' not in output and 'samtools (Tools for alignments in the SAM format)' in output:
        samtools_exe = 'samtools'
if not samtools_exe:
    raise MissingExternalDependencyError('Install samtools and correctly set the file path to the program\n        if you want to use it from Biopython')

class SamtoolsTestCase(unittest.TestCase):
    """Class for implementing Samtools test cases."""

    def setUp(self):
        if False:
            return 10
        self.files_to_clean = set()
        self.samfile1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SamBam', 'sam1.sam')
        self.reference = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BWA', 'human_g1k_v37_truncated.fasta')
        self.referenceindexfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BWA', 'human_g1k_v37_truncated.fasta.fai')
        self.samfile2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SamBam', 'sam2.sam')
        self.bamfile1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SamBam', 'bam1.bam')
        self.bamfile2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SamBam', 'bam2.bam')
        self.outsamfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SamBam', 'out.sam')
        self.outbamfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SamBam', 'out.bam')
        self.bamindexfile1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SamBam', 'bam1.bam.bai')
        self.sortedbamfile1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SamBam', 'bam1_sorted.bam')
        self.sortedbamfile2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SamBam', 'bam2_sorted.bam')
        self.files_to_clean = [self.referenceindexfile, self.bamindexfile1, self.outbamfile]

    def tearDown(self):
        if False:
            return 10
        for filename in self.files_to_clean:
            if os.path.isfile(filename):
                os.remove(filename)

    def test_view(self):
        if False:
            i = 10
            return i + 15
        'Test for samtools view.'
        cmdline = SamtoolsViewCommandline(samtools_exe)
        cmdline.set_parameter('input_file', self.bamfile1)
        (stdout_bam, stderr_bam) = cmdline()
        self.assertTrue(stderr_bam.startswith(''), f'SAM file viewing failed: \n{cmdline}\nStdout:{stdout_bam}')
        cmdline.set_parameter('input_file', self.samfile1)
        cmdline.set_parameter('S', True)
        (stdout_sam, stderr_sam) = cmdline()
        self.assertTrue(stdout_sam.startswith('HWI-1KL120:88:D0LRBACXX:1:1101:1780:2146'), f'SAM file  viewing failed:\n{cmdline}\nStderr:{stderr_sam}')

    def create_fasta_index(self):
        if False:
            for i in range(10):
                print('nop')
        'Create index for reference fasta sequence.'
        cmdline = SamtoolsFaidxCommandline(samtools_exe)
        cmdline.set_parameter('reference', self.reference)
        (stdout, stderr) = cmdline()

    def create_bam_index(self, input_bam):
        if False:
            return 10
        'Create index of an input bam file.'
        cmdline = SamtoolsIndexCommandline(samtools_exe)
        cmdline.set_parameter('input_bam', input_bam)
        (stdout, stderr) = cmdline()

    def test_faidx(self):
        if False:
            while True:
                i = 10
        cmdline = SamtoolsFaidxCommandline(samtools_exe)
        cmdline.set_parameter('reference', self.reference)
        (stdout, stderr) = cmdline()
        self.assertFalse(stderr, f'Samtools faidx failed:\n{cmdline}\nStderr:{stderr}')
        self.assertTrue(os.path.isfile(self.referenceindexfile))

    def test_calmd(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for samtools calmd.'
        self.create_fasta_index()
        cmdline = SamtoolsCalmdCommandline(samtools_exe)
        cmdline.set_parameter('reference', self.reference)
        cmdline.set_parameter('input_bam', self.bamfile1)
        if os.path.exists(self.referenceindexfile):
            stderr_calmd_expected = ''
        else:
            stderr_calmd_expected = '[fai_load] build FASTA index.\n'
        (stdout, stderr) = cmdline()
        self.assertEqual(stderr, stderr_calmd_expected)

    def test_cat(self):
        if False:
            return 10
        cmdline = SamtoolsCatCommandline(samtools_exe)
        cmdline.set_parameter('o', self.outbamfile)
        cmdline.set_parameter('input_bam', [self.bamfile1, self.bamfile2])
        (stdout, stderr) = cmdline()
        self.assertEqual(stderr, '')

    def test_sort(self):
        if False:
            for i in range(10):
                print('nop')
        cmdline = SamtoolsVersion0xSortCommandline(samtools_exe)
        cmdline.set_parameter('input', self.bamfile1)
        cmdline.set_parameter('out_prefix', 'SamBam/out')
        try:
            (stdout, stderr) = cmdline()
        except ApplicationError as err:
            if '[bam_sort] Use -T PREFIX / -o FILE to specify temporary and final output files' in str(err):
                cmdline = SamtoolsVersion1xSortCommandline(samtools_exe)
                cmdline.set_parameter('input', self.bamfile1)
                cmdline.set_parameter('-T', 'out')
                cmdline.set_parameter('-o', 'out.bam')
                try:
                    (stdout, stderr) = cmdline()
                except ApplicationError:
                    raise
            else:
                raise
        self.assertFalse(stderr, f'Samtools sort failed:\n{cmdline}\nStderr:{stderr}')

    def test_index(self):
        if False:
            for i in range(10):
                print('nop')
        cmdline = SamtoolsIndexCommandline(samtools_exe)
        cmdline.set_parameter('input_bam', self.bamfile1)
        (stdout, stderr) = cmdline()
        self.assertFalse(stderr, f'Samtools index failed:\n{cmdline}\nStderr:{stderr}')
        self.assertTrue(os.path.exists(self.bamindexfile1))

    def test_idxstats(self):
        if False:
            return 10
        self.create_bam_index(self.bamfile1)
        cmdline = SamtoolsIdxstatsCommandline(samtools_exe)
        cmdline.set_parameter('input_bam', self.bamfile1)
        (stdout, stderr) = cmdline()
        self.assertFalse(stderr, f'Samtools idxstats failed:\n{cmdline}\nStderr:{stderr}')

    def test_merge(self):
        if False:
            while True:
                i = 10
        cmdline = SamtoolsMergeCommandline(samtools_exe)
        cmdline.set_parameter('input_bam', [self.bamfile1, self.bamfile2])
        cmdline.set_parameter('out_bam', self.outbamfile)
        cmdline.set_parameter('f', True)
        (stdout, stderr) = cmdline()
        self.assertTrue(not stderr or stderr.strip() == '[W::bam_merge_core2] No @HD tag found.', f'Samtools merge failed:\n{cmdline}\nStderr:{stderr}')
        self.assertTrue(os.path.exists(self.outbamfile))

    def test_mpileup(self):
        if False:
            return 10
        cmdline = SamtoolsMpileupCommandline(samtools_exe)
        cmdline.set_parameter('input_file', [self.bamfile1])
        (stdout, stderr) = cmdline()
        self.assertNotIn('[bam_pileup_core]', stdout)

    def test_mpileup_list(self):
        if False:
            for i in range(10):
                print('nop')
        cmdline = SamtoolsMpileupCommandline(samtools_exe)
        cmdline.set_parameter('input_file', [self.sortedbamfile1, self.sortedbamfile2])
        (stdout, stderr) = cmdline()
        self.assertNotIn('[bam_pileup_core]', stdout)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)