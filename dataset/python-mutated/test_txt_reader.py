import unittest
from tempfile import NamedTemporaryFile
import os
import warnings
from Orange.data import Table, ContinuousVariable, DiscreteVariable
from Orange.data.io import CSVReader
from Orange.tests import test_filename
tab_file = 'Feature 1\tFeature 2\tFeature 3\n1.0      \t1.3        \t5\n2.0      \t42        \t7\n'
csv_file = 'Feature 1,   Feature 2,Feature 3\n1.0,      1.3,       5\n2.0,      42,        7\n'
tab_file_nh = '1.0      \t1.3        \t5\n2.0      \t42        \t7\n'
csv_file_nh = '1.0,      1.3,       5\n2.0,      42,        7\n'
noncont_marked_cont = 'a,b\nd,c\n,\ne,1\nf,g\n'
csv_file_missing = 'A,B\n1,A\n2,B\n3,A\n?,B\n5,?\n'

class TestTabReader(unittest.TestCase):

    def read_easy(self, s, name):
        if False:
            print('Hello World!')
        file = NamedTemporaryFile('wt', delete=False)
        filename = file.name
        try:
            file.write(s)
            file.close()
            table = CSVReader(filename).read()
            (f1, f2, f3) = table.domain.variables
            self.assertIsInstance(f1, DiscreteVariable)
            self.assertEqual(f1.name, name + '1')
            self.assertIsInstance(f2, ContinuousVariable)
            self.assertEqual(f2.name, name + '2')
            self.assertIsInstance(f3, ContinuousVariable)
            self.assertEqual(f3.name, name + '3')
        finally:
            os.remove(filename)

    def test_read_tab(self):
        if False:
            return 10
        self.read_easy(tab_file, 'Feature ')
        self.read_easy(tab_file_nh, 'Feature ')

    def test_read_csv(self):
        if False:
            print('Hello World!')
        self.read_easy(csv_file, 'Feature ')
        self.read_easy(csv_file_nh, 'Feature ')

    def test_read_csv_with_na(self):
        if False:
            print('Hello World!')
        with NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(csv_file_missing)
        table = CSVReader(tmp.name).read()
        os.unlink(tmp.name)
        (f1, f2) = table.domain.variables
        self.assertIsInstance(f1, ContinuousVariable)
        self.assertIsInstance(f2, DiscreteVariable)

    def test_read_nonutf8_encoding(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                Table(test_filename('datasets/invalid_characters.tab'))

    def test_noncontinous_marked_continuous(self):
        if False:
            return 10
        file = NamedTemporaryFile('wt', delete=False)
        file.write(noncont_marked_cont)
        file.close()
        with self.assertRaises(ValueError) as cm:
            table = CSVReader(file.name).read()
        self.assertIn('line 5, column 2', cm.exception.args[0])

    def test_pr1734(self):
        if False:
            i = 10
            return i + 15
        ContinuousVariable('foo')
        file = NamedTemporaryFile('wt', delete=False)
        filename = file.name
        try:
            file.write('foo\ntime\n\n123123123\n')
            file.close()
            CSVReader(filename).read()
        finally:
            os.remove(filename)

    def test_csv_sniffer(self):
        if False:
            print('Hello World!')
        reader = CSVReader(test_filename('datasets/test_asn_data_working.csv'))
        data = reader.read()
        self.assertEqual(len(data), 8)
        self.assertEqual(len(data.domain.variables) + len(data.domain.metas), 15)
if __name__ == '__main__':
    unittest.main()