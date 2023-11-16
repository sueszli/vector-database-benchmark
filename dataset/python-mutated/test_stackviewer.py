"""Test stackviewer, coverage 63%."""
from idlelib import stackviewer
import unittest
from test.support import requires
from tkinter import Tk
from idlelib.tree import TreeNode, ScrolledCanvas
import sys

class StackBrowserTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        svs = stackviewer.sys
        try:
            abc
        except NameError:
            (svs.last_type, svs.last_value, svs.last_traceback) = sys.exc_info()
        requires('gui')
        cls.root = Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        svs = stackviewer.sys
        del svs.last_traceback, svs.last_type, svs.last_value
        cls.root.update_idletasks()
        cls.root.destroy()
        del cls.root

    def test_init(self):
        if False:
            while True:
                i = 10
        sb = stackviewer.StackBrowser(self.root)
        isi = self.assertIsInstance
        isi(stackviewer.sc, ScrolledCanvas)
        isi(stackviewer.item, stackviewer.StackTreeItem)
        isi(stackviewer.node, TreeNode)
if __name__ == '__main__':
    unittest.main(verbosity=2)