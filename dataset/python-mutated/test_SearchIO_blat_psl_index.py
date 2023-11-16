"""Tests for SearchIO blat-psl indexing."""
import os
import unittest
from search_tests_common import CheckIndex

class BlatPslIndexCases(CheckIndex):
    fmt = 'blat-psl'

    def test_psl_34_001(self):
        if False:
            for i in range(10):
                print('nop')
        'Test blat-psl indexing, multiple queries.'
        filename = os.path.join('Blat', 'psl_34_001.psl')
        self.check_index(filename, self.fmt)

    def test_psl_34_002(self):
        if False:
            print('Hello World!')
        'Test blat-psl indexing, single query, no hits.'
        filename = os.path.join('Blat', 'psl_34_002.psl')
        self.check_index(filename, self.fmt)

    def test_psl_34_003(self):
        if False:
            while True:
                i = 10
        'Test blat-psl indexing, single query, single hit.'
        filename = os.path.join('Blat', 'psl_34_003.psl')
        self.check_index(filename, self.fmt)

    def test_psl_34_004(self):
        if False:
            for i in range(10):
                print('nop')
        'Test blat-psl indexing, single query, multiple hits with multiple hsps.'
        filename = os.path.join('Blat', 'psl_34_004.psl')
        self.check_index(filename, self.fmt)

    def test_psl_34_005(self):
        if False:
            print('Hello World!')
        'Test blat-psl indexing, multiple queries, no header.'
        filename = os.path.join('Blat', 'psl_34_005.psl')
        self.check_index(filename, self.fmt)

    def test_psl_34_006(self):
        if False:
            for i in range(10):
                print('nop')
        'Test blat-pslx indexing, multiple queries.'
        filename = os.path.join('Blat', 'pslx_34_001.pslx')
        self.check_index(filename, self.fmt, pslx=True)

    def test_psl_34_007(self):
        if False:
            print('Hello World!')
        'Test blat-pslx indexing, single query, no hits.'
        filename = os.path.join('Blat', 'pslx_34_002.pslx')
        self.check_index(filename, self.fmt, pslx=True)

    def test_psl_34_008(self):
        if False:
            while True:
                i = 10
        'Test blat-pslx indexing, single query, single hit.'
        filename = os.path.join('Blat', 'pslx_34_003.pslx')
        self.check_index(filename, self.fmt, pslx=True)

    def test_psl_34_009(self):
        if False:
            while True:
                i = 10
        'Test blat-pslx indexing, single query, multiple hits with multiple hsps.'
        filename = os.path.join('Blat', 'pslx_34_004.pslx')
        self.check_index(filename, self.fmt, pslx=True)

    def test_psl_34_010(self):
        if False:
            while True:
                i = 10
        'Test blat-pslx indexing, multiple queries, no header.'
        filename = os.path.join('Blat', 'pslx_34_005.pslx')
        self.check_index(filename, self.fmt, pslx=True)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)