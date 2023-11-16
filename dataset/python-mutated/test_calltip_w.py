"""Test calltip_w, coverage 18%."""
from idlelib import calltip_w
import unittest
from test.support import requires
from tkinter import Tk, Text

class CallTipWindowTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        requires('gui')
        cls.root = Tk()
        cls.root.withdraw()
        cls.text = Text(cls.root)
        cls.calltip = calltip_w.CalltipWindow(cls.text)

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        cls.root.update_idletasks()
        cls.root.destroy()
        del cls.text, cls.root

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.calltip.anchor_widget, self.text)
if __name__ == '__main__':
    unittest.main(verbosity=2)