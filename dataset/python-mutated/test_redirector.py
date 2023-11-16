"""Test redirector, coverage 100%."""
from idlelib.redirector import WidgetRedirector
import unittest
from test.support import requires
from tkinter import Tk, Text, TclError
from idlelib.idle_test.mock_idle import Func

class InitCloseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        requires('gui')
        cls.root = Tk()
        cls.root.withdraw()
        cls.text = Text(cls.root)

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        del cls.text
        cls.root.destroy()
        del cls.root

    def test_init(self):
        if False:
            i = 10
            return i + 15
        redir = WidgetRedirector(self.text)
        self.assertEqual(redir.widget, self.text)
        self.assertEqual(redir.tk, self.text.tk)
        self.assertRaises(TclError, WidgetRedirector, self.text)
        redir.close()

    def test_close(self):
        if False:
            i = 10
            return i + 15
        redir = WidgetRedirector(self.text)
        redir.register('insert', Func)
        redir.close()
        self.assertEqual(redir._operations, {})
        self.assertFalse(hasattr(self.text, 'widget'))

class WidgetRedirectorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        requires('gui')
        cls.root = Tk()
        cls.root.withdraw()
        cls.text = Text(cls.root)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        del cls.text
        cls.root.update_idletasks()
        cls.root.destroy()
        del cls.root

    def setUp(self):
        if False:
            print('Hello World!')
        self.redir = WidgetRedirector(self.text)
        self.func = Func()
        self.orig_insert = self.redir.register('insert', self.func)
        self.text.insert('insert', 'asdf')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.text.delete('1.0', 'end')
        self.redir.close()

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        self.assertIn('Redirector', repr(self.redir))
        self.assertIn('Original', repr(self.orig_insert))

    def test_register(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.text.get('1.0', 'end'), '\n')
        self.assertEqual(self.func.args, ('insert', 'asdf'))
        self.assertIn('insert', self.redir._operations)
        self.assertIn('insert', self.text.__dict__)
        self.assertEqual(self.text.insert, self.func)

    def test_original_command(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.orig_insert.operation, 'insert')
        self.assertEqual(self.orig_insert.tk_call, self.text.tk.call)
        self.orig_insert('insert', 'asdf')
        self.assertEqual(self.text.get('1.0', 'end'), 'asdf\n')

    def test_unregister(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(self.redir.unregister('invalid operation name'))
        self.assertEqual(self.redir.unregister('insert'), self.func)
        self.assertNotIn('insert', self.redir._operations)
        self.assertNotIn('insert', self.text.__dict__)

    def test_unregister_no_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        del self.text.insert
        self.assertEqual(self.redir.unregister('insert'), self.func)

    def test_dispatch_intercept(self):
        if False:
            print('Hello World!')
        self.func.__init__(True)
        self.assertTrue(self.redir.dispatch('insert', False))
        self.assertFalse(self.func.args[0])

    def test_dispatch_bypass(self):
        if False:
            while True:
                i = 10
        self.orig_insert('insert', 'asdf')
        self.assertEqual(self.redir.dispatch('delete', '1.0', 'end'), '')
        self.assertEqual(self.text.get('1.0', 'end'), '\n')

    def test_dispatch_error(self):
        if False:
            print('Hello World!')
        self.func.__init__(TclError())
        self.assertEqual(self.redir.dispatch('insert', False), '')
        self.assertEqual(self.redir.dispatch('invalid'), '')

    def test_command_dispatch(self):
        if False:
            print('Hello World!')
        self.root.call(self.text._w, 'insert', 'hello')
        self.assertEqual(self.func.args, ('hello',))
        self.assertEqual(self.text.get('1.0', 'end'), '\n')
        self.func.__init__(TclError())
        self.assertEqual(self.root.call(self.text._w, 'insert', 'boo'), '')
if __name__ == '__main__':
    unittest.main(verbosity=2)