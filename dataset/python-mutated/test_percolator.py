"""Test percolator, coverage 100%."""
from idlelib.percolator import Percolator, Delegator
import unittest
from test.support import requires
requires('gui')
from tkinter import Text, Tk, END

class MyFilter(Delegator):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        Delegator.__init__(self, None)

    def insert(self, *args):
        if False:
            i = 10
            return i + 15
        self.insert_called_with = args
        self.delegate.insert(*args)

    def delete(self, *args):
        if False:
            while True:
                i = 10
        self.delete_called_with = args
        self.delegate.delete(*args)

    def uppercase_insert(self, index, chars, tags=None):
        if False:
            i = 10
            return i + 15
        chars = chars.upper()
        self.delegate.insert(index, chars)

    def lowercase_insert(self, index, chars, tags=None):
        if False:
            return 10
        chars = chars.lower()
        self.delegate.insert(index, chars)

    def dont_insert(self, index, chars, tags=None):
        if False:
            for i in range(10):
                print('nop')
        pass

class PercolatorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.root = Tk()
        cls.text = Text(cls.root)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        del cls.text
        cls.root.destroy()
        del cls.root

    def setUp(self):
        if False:
            while True:
                i = 10
        self.percolator = Percolator(self.text)
        self.filter_one = MyFilter()
        self.filter_two = MyFilter()
        self.percolator.insertfilter(self.filter_one)
        self.percolator.insertfilter(self.filter_two)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.percolator.close()
        self.text.delete('1.0', END)

    def test_insertfilter(self):
        if False:
            return 10
        self.assertIsNotNone(self.filter_one.delegate)
        self.assertEqual(self.percolator.top, self.filter_two)
        self.assertEqual(self.filter_two.delegate, self.filter_one)
        self.assertEqual(self.filter_one.delegate, self.percolator.bottom)

    def test_removefilter(self):
        if False:
            return 10
        filter_three = MyFilter()
        self.percolator.removefilter(self.filter_two)
        self.assertEqual(self.percolator.top, self.filter_one)
        self.assertIsNone(self.filter_two.delegate)
        filter_three = MyFilter()
        self.percolator.insertfilter(self.filter_two)
        self.percolator.insertfilter(filter_three)
        self.percolator.removefilter(self.filter_one)
        self.assertEqual(self.percolator.top, filter_three)
        self.assertEqual(filter_three.delegate, self.filter_two)
        self.assertEqual(self.filter_two.delegate, self.percolator.bottom)
        self.assertIsNone(self.filter_one.delegate)

    def test_insert(self):
        if False:
            while True:
                i = 10
        self.text.insert('insert', 'foo')
        self.assertEqual(self.text.get('1.0', END), 'foo\n')
        self.assertTupleEqual(self.filter_one.insert_called_with, ('insert', 'foo', None))

    def test_modify_insert(self):
        if False:
            i = 10
            return i + 15
        self.filter_one.insert = self.filter_one.uppercase_insert
        self.text.insert('insert', 'bAr')
        self.assertEqual(self.text.get('1.0', END), 'BAR\n')

    def test_modify_chain_insert(self):
        if False:
            print('Hello World!')
        filter_three = MyFilter()
        self.percolator.insertfilter(filter_three)
        self.filter_two.insert = self.filter_two.uppercase_insert
        self.filter_one.insert = self.filter_one.lowercase_insert
        self.text.insert('insert', 'BaR')
        self.assertEqual(self.text.get('1.0', END), 'bar\n')

    def test_dont_insert(self):
        if False:
            return 10
        self.filter_one.insert = self.filter_one.dont_insert
        self.text.insert('insert', 'foo bar')
        self.assertEqual(self.text.get('1.0', END), '\n')
        self.filter_one.insert = self.filter_one.dont_insert
        self.text.insert('insert', 'foo bar')
        self.assertEqual(self.text.get('1.0', END), '\n')

    def test_without_filter(self):
        if False:
            for i in range(10):
                print('nop')
        self.text.insert('insert', 'hello')
        self.assertEqual(self.text.get('1.0', 'end'), 'hello\n')

    def test_delete(self):
        if False:
            return 10
        self.text.insert('insert', 'foo')
        self.text.delete('1.0', '1.2')
        self.assertEqual(self.text.get('1.0', END), 'o\n')
        self.assertTupleEqual(self.filter_one.delete_called_with, ('1.0', '1.2'))
if __name__ == '__main__':
    unittest.main(verbosity=2)