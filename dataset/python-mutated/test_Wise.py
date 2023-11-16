"""Tests for Wise module."""
import doctest
import random
import sys
import unittest
import warnings
from subprocess import getoutput
from io import StringIO
from Bio import BiopythonDeprecationWarning, MissingExternalDependencyError
if sys.platform == 'win32':
    raise MissingExternalDependencyError("Don't know how to find the Wise2 tool dnal on Windows.")
not_found_types = ['command not found', 'dnal: not found', 'not recognized']
dnal_output = getoutput('dnal')
for not_found in not_found_types:
    if not_found in dnal_output:
        raise MissingExternalDependencyError('Install Wise2 (dnal) if you want to use Bio.Wise.')
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=BiopythonDeprecationWarning)
    from Bio import Wise
    from Bio.Wise import psw

class TestWiseDryRun(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.old_stdout = sys.stdout
        sys.stdout = StringIO()

    def test_dnal(self):
        if False:
            for i in range(10):
                print('nop')
        'Call dnal, and do a trivial check on its output.'
        Wise.align(['dnal'], ('seq1.fna', 'seq2.fna'), kbyte=100000, dry_run=True)
        output = sys.stdout.getvalue().replace(' -quiet ', ' ')
        self.assertTrue(output.startswith('dnal -kbyte 100000 seq1.fna seq2.fna'), output[:200])

    def test_psw(self):
        if False:
            for i in range(10):
                print('nop')
        'Call psw, and do a trivial check on its output.'
        Wise.align(['psw'], ('seq1.faa', 'seq2.faa'), dry_run=True, kbyte=4)
        output = sys.stdout.getvalue().replace(' -quiet ', ' ')
        self.assertTrue(output.startswith('psw -kbyte 4 seq1.faa seq2.faa'), output[:200])

    def tearDown(self):
        if False:
            print('Hello World!')
        sys.stdout = self.old_stdout

class TestWise(unittest.TestCase):

    def test_align(self):
        if False:
            while True:
                i = 10
        'Call dnal with optional arguments, and do a trivial check on the output.'
        temp_file = Wise.align(['dnal'], ('Wise/human_114_g01_exons.fna_01', 'Wise/human_114_g02_exons.fna_01'), kbyte=100000, force_type='DNA', quiet=True)
        line = temp_file.readline().rstrip()
        if line == 'Score 114':
            line = temp_file.readline().rstrip()
        if line == 'ENSG00000172135   AGGGAAAGCCCCTAAGCTC--CTGATCTATGCTGCATCCAGTTTGCAAAGTGGGGTCCC':
            pass
        elif line == 'ENSG00000172135   AGGGAAAGCCCCTAAGCTC--CTGATCTATGCTGCATCCAGTTTGCAAAG-TGGGGTCC':
            pass
        else:
            self.fail(line)

class TestPSW(unittest.TestCase):

    def test_Alignment_normal(self):
        if False:
            for i in range(10):
                print('nop')
        a = psw.Alignment()
        a.append(psw.ColumnUnit(0, 98, 'SEQUENCE'))
        a.append(psw.ColumnUnit(1, 200, 'SEQUENCE'))
        a.append(psw.ColumnUnit(0, 98, 'INSERT'))
        a.append(psw.ColumnUnit(1, 201, 'SEQUENCE'))
        a.append(psw.ColumnUnit(0, 98, 'END'))
        a.append(psw.ColumnUnit(1, 201, 'END'))
        self.assertEqual(str(a), '[SEQUENCE(98, 200), INSERT(98, 201), END(98, 201)]')

    def test_Alignment_assertions(self):
        if False:
            for i in range(10):
                print('nop')
        a = psw.Alignment()
        self.assertRaises(AssertionError, a.append, psw.ColumnUnit(1, 200, 'SEQUENCE'))
        a.append(psw.ColumnUnit(0, 98, 'SEQUENCE'))
        self.assertRaises(AssertionError, a.append, psw.ColumnUnit(0, 200, 'SEQUENCE'))
        a.append(psw.ColumnUnit(1, 200, 'SEQUENCE'))
        self.assertRaises(AssertionError, a.append, psw.ColumnUnit(1, 200, 'SEQUENCE'))

    def test_AlignmentColumn_kinds(self):
        if False:
            return 10
        ac = psw.AlignmentColumn(psw.ColumnUnit(0, random.randint(0, 9999), 'SEQUENCE'))
        ac.append(psw.ColumnUnit(1, random.randint(0, 9999), 'INSERT'))
        self.assertEqual(ac.kind, 'INSERT')
        ac = psw.AlignmentColumn(psw.ColumnUnit(0, random.randint(0, 9999), 'INSERT'))
        ac.append(psw.ColumnUnit(1, random.randint(0, 9999), 'SEQUENCE'))
        self.assertEqual(ac.kind, 'INSERT')
        ac = psw.AlignmentColumn(psw.ColumnUnit(0, random.randint(0, 9999), 'SEQUENCE'))
        ac.append(psw.ColumnUnit(1, random.randint(0, 9999), 'SEQUENCE'))
        self.assertEqual(ac.kind, 'SEQUENCE')
        ac = psw.AlignmentColumn(psw.ColumnUnit(0, random.randint(0, 9999), 'SEQUENCE'))
        ac.append(psw.ColumnUnit(1, random.randint(0, 9999), 'END'))
        self.assertEqual(ac.kind, 'END')

    def test_AlignmentColumn_repr(self):
        if False:
            return 10
        ac = psw.AlignmentColumn(psw.ColumnUnit(0, 34, 'SEQUENCE'))
        ac.append(psw.ColumnUnit(1, 55, 'END'))
        self.assertEqual(repr(ac), 'END(34, 55)')

    def test_AlignmentColumn_full(self):
        if False:
            return 10
        ac = psw.AlignmentColumn(psw.ColumnUnit(0, random.randint(0, 9999), 'SEQUENCE'))
        ac.append(psw.ColumnUnit(1, random.randint(0, 9999), 'END'))
        self.assertRaises(psw.AlignmentColumnFullException, ac.append, psw.ColumnUnit(1, random.randint(0, 9999), 'END'))

    def test_AlignmentColumn_assertions(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(AssertionError, psw.AlignmentColumn, psw.ColumnUnit(1, random.randint(0, 9999), 'SEQUENCE'))
        ac = psw.AlignmentColumn(psw.ColumnUnit(0, random.randint(0, 9999), 'SEQUENCE'))
        self.assertRaises(AssertionError, ac.append, psw.ColumnUnit(0, random.randint(0, 9999), 'SEQUENCE'))

    def test_ColumnUnit(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(repr(psw.ColumnUnit(0, 33, 'SEQUENCE')), "ColumnUnit(unit=0, column=33, kind='SEQUENCE')")
        self.assertEqual(repr(psw.ColumnUnit(1, 33, 'INSERT')), "ColumnUnit(unit=1, column=33, kind='INSERT')")
        self.assertEqual(repr(psw.ColumnUnit(1, 33, 'END')), "ColumnUnit(unit=1, column=33, kind='END')")
    PARSED = '[SEQUENCE(39, 22), SEQUENCE(40, 23), SEQUENCE(41, 24), SEQUENCE(42, 25), SEQUENCE(43, 26), SEQUENCE(44, 27), END(0, 27)]'

def run_tests(argv):
    if False:
        print('Hello World!')
    test_suite = testing_suite()
    runner = unittest.TextTestRunner(sys.stdout, verbosity=2)
    runner.run(test_suite)

def testing_suite():
    if False:
        for i in range(10):
            print('nop')
    'Generate the suite of tests.'
    unittest_suite = unittest.TestSuite()
    test_loader = unittest.TestLoader()
    test_loader.testMethodPrefix = 'test_'
    tests = [TestPSW]
    for test in tests:
        cur_suite = test_loader.loadTestsFromTestCase(test)
        unittest_suite.addTest(cur_suite)
    doctest_suite = doctest.DocTestSuite(psw)
    big_suite = unittest.TestSuite((unittest_suite, doctest_suite))
    return big_suite
if __name__ == '__main__':
    unittest_suite = unittest.TestLoader().loadTestsFromName('test_Wise')
    wise_doctest_suite = doctest.DocTestSuite(Wise)
    psw_doctest_suite = doctest.DocTestSuite(psw)
    suite = unittest.TestSuite((unittest_suite, wise_doctest_suite, psw_doctest_suite))
    runner = unittest.TextTestRunner(sys.stdout, verbosity=2)
    runner.run(suite)