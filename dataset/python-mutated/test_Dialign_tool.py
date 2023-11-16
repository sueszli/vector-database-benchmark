"""Unittests for Bio.Align.Applications interface for DIALIGN2-2."""
import sys
import os
import unittest
from Bio import MissingExternalDependencyError
from Bio.Align.Applications import DialignCommandline
os.environ['LANG'] = 'C'
dialign_exe = None
if sys.platform == 'win32':
    raise MissingExternalDependencyError('DIALIGN2-2 not available on Windows')
else:
    from subprocess import getoutput
    output = getoutput('dialign2-2')
    if 'not found' not in output and 'not recognized' not in output:
        if 'dialign2-2' in output.lower():
            dialign_exe = 'dialign2-2'
            if 'DIALIGN2_DIR' not in os.environ:
                raise MissingExternalDependencyError('Environment variable DIALIGN2_DIR for DIALIGN2-2 missing.')
            if not os.path.isdir(os.environ['DIALIGN2_DIR']):
                raise MissingExternalDependencyError('Environment variable DIALIGN2_DIR for DIALIGN2-2 is not a valid directory.')
            if not os.path.isfile(os.path.join(os.environ['DIALIGN2_DIR'], 'BLOSUM')):
                raise MissingExternalDependencyError('Environment variable DIALIGN2_DIR directory missing BLOSUM file.')
if not dialign_exe:
    raise MissingExternalDependencyError('Install DIALIGN2-2 if you want to use the Bio.Align.Applications wrapper.')

class DialignApplication(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.infile1 = 'Fasta/f002'
        self.outfile1 = 'Fasta/f002.ali'
        self.outfile2 = 'Fasta/f002.ms'

    def tearDown(self):
        if False:
            print('Hello World!')
        if os.path.isfile(self.outfile1):
            os.remove(self.outfile1)
        if os.path.isfile(self.outfile2):
            os.remove(self.outfile2)

    def test_Dialign_simple(self):
        if False:
            print('Hello World!')
        'Simple round-trip through app with infile.'
        cmdline = DialignCommandline(dialign_exe, input=self.infile1)
        self.assertEqual(str(cmdline), dialign_exe + ' Fasta/f002')
        (stdout, stderr) = cmdline()
        self.assertEqual(stderr, '')
        self.assertEqual(stdout, '')
        self.assertTrue(os.path.exists(self.outfile1))

    def test_Dialign_simple_with_options(self):
        if False:
            i = 10
            return i + 15
        'Simple round-trip through app with infile and options.'
        cmdline = DialignCommandline(dialign_exe)
        cmdline.set_parameter('input', self.infile1)
        cmdline.set_parameter('-max_link', True)
        cmdline.set_parameter('stars', 4)
        self.assertEqual(str(cmdline), dialign_exe + ' -max_link -stars 4 Fasta/f002')
        (stdout, stderr) = cmdline()
        self.assertEqual(stderr, '')
        self.assertEqual(stdout, '')
        self.assertTrue(os.path.exists(self.outfile1))

    def test_Dialign_simple_with_MSF_output(self):
        if False:
            return 10
        'Simple round-trip through app with infile, output MSF.'
        cmdline = DialignCommandline(dialign_exe)
        cmdline.input = self.infile1
        cmdline.msf = True
        self.assertEqual(str(cmdline), dialign_exe + ' -msf Fasta/f002')
        (stdout, stderr) = cmdline()
        self.assertEqual(stderr, '')
        self.assertEqual(stdout, '')
        self.assertTrue(os.path.exists(self.outfile1))
        self.assertTrue(os.path.exists(self.outfile2))

    def test_Dialign_complex_command_line(self):
        if False:
            while True:
                i = 10
        'Round-trip through app with complex command line.'
        cmdline = DialignCommandline(dialign_exe)
        cmdline.set_parameter('input', self.infile1)
        cmdline.set_parameter('-nt', True)
        cmdline.set_parameter('-thr', 4)
        cmdline.set_parameter('stars', 9)
        cmdline.set_parameter('-ow', True)
        cmdline.set_parameter('mask', True)
        cmdline.set_parameter('-cs', True)
        self.assertEqual(str(cmdline), dialign_exe + ' -cs -mask -nt -ow -stars 9 -thr 4 Fasta/f002')
        (stdout, stderr) = cmdline()
        self.assertEqual(stderr, '')
        self.assertTrue(os.path.exists(self.outfile1))
        self.assertTrue(stdout.startswith(' e_len = 633'))
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)