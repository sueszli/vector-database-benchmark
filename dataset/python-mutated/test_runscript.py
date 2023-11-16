"""Test runscript, coverage 16%."""
from idlelib import runscript
import unittest
from test.support import requires
from tkinter import Tk
from idlelib.editor import EditorWindow

class ScriptBindingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        requires('gui')
        cls.root = Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        cls.root.update_idletasks()
        for id in cls.root.tk.call('after', 'info'):
            cls.root.after_cancel(id)
        cls.root.destroy()
        del cls.root

    def test_init(self):
        if False:
            i = 10
            return i + 15
        ew = EditorWindow(root=self.root)
        sb = runscript.ScriptBinding(ew)
        ew._close()
if __name__ == '__main__':
    unittest.main(verbosity=2)