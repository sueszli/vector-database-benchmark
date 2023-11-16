"""Test scrolledlist, coverage 38%."""
from idlelib.scrolledlist import ScrolledList
import unittest
from test.support import requires
requires('gui')
from tkinter import Tk

class ScrolledListTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls.root = Tk()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls.root.destroy()
        del cls.root

    def test_init(self):
        if False:
            return 10
        ScrolledList(self.root)
if __name__ == '__main__':
    unittest.main(verbosity=2)