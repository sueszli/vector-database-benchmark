"""Test searchbase, coverage 98%."""
import unittest
from test.support import requires
from tkinter import Text, Tk, Toplevel
from tkinter.ttk import Frame
from idlelib import searchengine as se
from idlelib import searchbase as sdb
from idlelib.idle_test.mock_idle import Func

class SearchDialogBaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        requires('gui')
        cls.root = Tk()

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        cls.root.update_idletasks()
        cls.root.destroy()
        del cls.root

    def setUp(self):
        if False:
            return 10
        self.engine = se.SearchEngine(self.root)
        self.dialog = sdb.SearchDialogBase(root=self.root, engine=self.engine)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.dialog.close()

    def test_open_and_close(self):
        if False:
            while True:
                i = 10
        self.dialog.default_command = None
        toplevel = Toplevel(self.root)
        text = Text(toplevel)
        self.dialog.open(text)
        self.assertEqual(self.dialog.top.state(), 'normal')
        self.dialog.close()
        self.assertEqual(self.dialog.top.state(), 'withdrawn')
        self.dialog.open(text, searchphrase='hello')
        self.assertEqual(self.dialog.ent.get(), 'hello')
        toplevel.update_idletasks()
        toplevel.destroy()

    def test_create_widgets(self):
        if False:
            print('Hello World!')
        self.dialog.create_entries = Func()
        self.dialog.create_option_buttons = Func()
        self.dialog.create_other_buttons = Func()
        self.dialog.create_command_buttons = Func()
        self.dialog.default_command = None
        self.dialog.create_widgets()
        self.assertTrue(self.dialog.create_entries.called)
        self.assertTrue(self.dialog.create_option_buttons.called)
        self.assertTrue(self.dialog.create_other_buttons.called)
        self.assertTrue(self.dialog.create_command_buttons.called)

    def test_make_entry(self):
        if False:
            return 10
        equal = self.assertEqual
        self.dialog.row = 0
        self.dialog.frame = Frame(self.root)
        (entry, label) = self.dialog.make_entry('Test:', 'hello')
        equal(label['text'], 'Test:')
        self.assertIn(entry.get(), 'hello')
        egi = entry.grid_info()
        equal(int(egi['row']), 0)
        equal(int(egi['column']), 1)
        equal(int(egi['rowspan']), 1)
        equal(int(egi['columnspan']), 1)
        equal(self.dialog.row, 1)

    def test_create_entries(self):
        if False:
            for i in range(10):
                print('nop')
        self.dialog.frame = Frame(self.root)
        self.dialog.row = 0
        self.engine.setpat('hello')
        self.dialog.create_entries()
        self.assertIn(self.dialog.ent.get(), 'hello')

    def test_make_frame(self):
        if False:
            i = 10
            return i + 15
        self.dialog.row = 0
        self.dialog.frame = Frame(self.root)
        (frame, label) = self.dialog.make_frame()
        self.assertEqual(label, '')
        self.assertEqual(str(type(frame)), "<class 'tkinter.ttk.Frame'>")
        (frame, label) = self.dialog.make_frame('testlabel')
        self.assertEqual(label['text'], 'testlabel')

    def btn_test_setup(self, meth):
        if False:
            i = 10
            return i + 15
        self.dialog.frame = Frame(self.root)
        self.dialog.row = 0
        return meth()

    def test_create_option_buttons(self):
        if False:
            print('Hello World!')
        e = self.engine
        for state in (0, 1):
            for var in (e.revar, e.casevar, e.wordvar, e.wrapvar):
                var.set(state)
            (frame, options) = self.btn_test_setup(self.dialog.create_option_buttons)
            for (spec, button) in zip(options, frame.pack_slaves()):
                (var, label) = spec
                self.assertEqual(button['text'], label)
                self.assertEqual(var.get(), state)

    def test_create_other_buttons(self):
        if False:
            return 10
        for state in (False, True):
            var = self.engine.backvar
            var.set(state)
            (frame, others) = self.btn_test_setup(self.dialog.create_other_buttons)
            buttons = frame.pack_slaves()
            for (spec, button) in zip(others, buttons):
                (val, label) = spec
                self.assertEqual(button['text'], label)
                if val == state:
                    self.assertEqual(var.get(), state)

    def test_make_button(self):
        if False:
            while True:
                i = 10
        self.dialog.frame = Frame(self.root)
        self.dialog.buttonframe = Frame(self.dialog.frame)
        btn = self.dialog.make_button('Test', self.dialog.close)
        self.assertEqual(btn['text'], 'Test')

    def test_create_command_buttons(self):
        if False:
            for i in range(10):
                print('nop')
        self.dialog.frame = Frame(self.root)
        self.dialog.create_command_buttons()
        closebuttoncommand = ''
        for child in self.dialog.buttonframe.winfo_children():
            if child['text'] == 'Close':
                closebuttoncommand = child['command']
        self.assertIn('close', closebuttoncommand)
if __name__ == '__main__':
    unittest.main(verbosity=2, exit=2)