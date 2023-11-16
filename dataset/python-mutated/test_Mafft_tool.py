"""Unittests for Bio.Align.Applications interface for MAFFT."""
import sys
import os
import unittest
import subprocess
from Bio import MissingExternalDependencyError
from Bio.Align.Applications import MafftCommandline
os.environ['LANG'] = 'C'
mafft_exe = None
if sys.platform == 'win32':
    raise MissingExternalDependencyError('Testing with MAFFT not implemented on Windows yet')
else:
    from subprocess import getoutput
    output = getoutput('mafft -help')
    if 'not found' not in output and 'not recognized' not in output:
        if 'MAFFT' in output:
            mafft_exe = 'mafft'
if not mafft_exe:
    raise MissingExternalDependencyError('Install MAFFT if you want to use the Bio.Align.Applications wrapper.')

def check_mafft_version(mafft_exe):
    if False:
        i = 10
        return i + 15
    child = subprocess.Popen(f'{mafft_exe} --help', stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
    (stdoutdata, stderrdata) = child.communicate()
    output = stdoutdata + '\n' + stderrdata
    return_code = child.returncode
    del child
    if 'correctly installed?' in output or 'mafft binaries have to be installed' in output:
        raise MissingExternalDependencyError('MAFFT does not seem to be correctly installed.')
    for marker in ['MAFFT version', 'MAFFT v']:
        index = output.find(marker)
        if index == -1:
            continue
        version = output[index + len(marker):].strip().split(None, 1)[0]
        major = int(version.split('.', 1)[0])
        if major < 6:
            raise MissingExternalDependencyError(f'Test requires MAFFT v6 or later (found {version}).')
        return (major, version)
    raise MissingExternalDependencyError("Couldn't determine MAFFT version.")
(version_major, version_string) = check_mafft_version(mafft_exe)

class MafftApplication(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.infile1 = 'Fasta/f002'

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if os.path.isfile('Fasta/f002.tree'):
            os.remove('Fasta/f002.tree')

    def test_Mafft_simple(self):
        if False:
            i = 10
            return i + 15
        'Simple round-trip through app with infile, result passed to stdout.'
        cmdline = MafftCommandline(mafft_exe, input=self.infile1)
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        (stdoutdata, stderrdata) = cmdline()
        self.assertTrue(stdoutdata.startswith('>gi|1348912|gb|G26680|G26680'))
        self.assertTrue('Progressive alignment ...' in stderrdata or 'Progressive alignment 1/' in stderrdata, stderrdata)
        self.assertNotIn('$#=0', stderrdata)

    def test_Mafft_with_options(self):
        if False:
            return 10
        'Simple round-trip through app with infile and options, result passed to stdout.'
        cmdline = MafftCommandline(mafft_exe)
        cmdline.set_parameter('input', self.infile1)
        cmdline.set_parameter('maxiterate', 100)
        cmdline.set_parameter('--localpair', True)
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        (stdoutdata, stderrdata) = cmdline()
        self.assertTrue(stdoutdata.startswith('>gi|1348912|gb|G26680|G26680'))
        self.assertNotIn('$#=0', stderrdata)

    def test_Mafft_with_Clustalw_output(self):
        if False:
            i = 10
            return i + 15
        'Simple round-trip through app with clustal output.'
        cmdline = MafftCommandline(mafft_exe)
        cmdline.input = self.infile1
        cmdline.clustalout = True
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        (stdoutdata, stderrdata) = cmdline()
        self.assertTrue(stdoutdata.startswith('CLUSTAL'), stdoutdata)
        self.assertNotIn('$#=0', stderrdata)
    if version_major >= 7:

        def test_Mafft_with_PHYLIP_output(self):
            if False:
                while True:
                    i = 10
            'Simple round-trip through app with PHYLIP output.'
            cmdline = MafftCommandline(mafft_exe, input=self.infile1, phylipout=True)
            self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
            (stdoutdata, stderrdata) = cmdline()
            self.assertTrue(stdoutdata.startswith((' 3 68', ' 3 69', ' 3 70')), stdoutdata)
            self.assertIn('gi|1348912 ', stdoutdata, stdoutdata)
            self.assertNotIn('gi|1348912|gb|G26680|G26680', stdoutdata, stdoutdata)
            self.assertNotIn('$#=0', stderrdata)

        def test_Mafft_with_PHYLIP_namelength(self):
            if False:
                return 10
            'Check PHYLIP with --namelength.'
            cmdline = MafftCommandline(mafft_exe, input=self.infile1, phylipout=True, namelength=50)
            self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
            (stdoutdata, stderrdata) = cmdline()
            self.assertTrue(stdoutdata.startswith((' 3 68', ' 3 69', ' 3 70')), stdoutdata)
            self.assertIn('gi|1348912|gb|G26680|G26680', stdoutdata, stdoutdata)
            self.assertNotIn('$#=0', stderrdata)

    def test_Mafft_with_complex_command_line(self):
        if False:
            while True:
                i = 10
        'Round-trip with complex command line.'
        cmdline = MafftCommandline(mafft_exe)
        cmdline.set_parameter('input', self.infile1)
        cmdline.set_parameter('--localpair', True)
        cmdline.set_parameter('--weighti', 4.2)
        cmdline.set_parameter('retree', 5)
        cmdline.set_parameter('maxiterate', 200)
        cmdline.set_parameter('--nofft', True)
        cmdline.set_parameter('op', 2.04)
        cmdline.set_parameter('--ep', 0.51)
        cmdline.set_parameter('--lop', 0.233)
        cmdline.set_parameter('lep', 0.2)
        cmdline.set_parameter('--reorder', True)
        cmdline.set_parameter('--treeout', True)
        cmdline.set_parameter('nuc', True)
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        self.assertEqual(str(cmdline), mafft_exe + ' --localpair --weighti 4.2 --retree 5 ' + '--maxiterate 200 --nofft --op 2.04 --ep 0.51' + ' --lop 0.233 --lep 0.2 --reorder --treeout' + ' --nuc Fasta/f002')
        (stdoutdata, stderrdata) = cmdline()
        self.assertTrue(stdoutdata.startswith('>gi|1348912|gb|G26680|G26680'))
        self.assertNotIn('$#=0', stderrdata)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)