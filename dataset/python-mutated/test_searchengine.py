"""Test searchengine, coverage 99%."""
from idlelib import searchengine as se
import unittest
from tkinter import BooleanVar, StringVar, TclError
from tkinter import messagebox
from idlelib.idle_test.mock_tk import Var, Mbox
from idlelib.idle_test.mock_tk import Text as mockText
import re

def setUpModule():
    if False:
        print('Hello World!')
    se.BooleanVar = Var
    se.StringVar = Var
    se.messagebox = Mbox

def tearDownModule():
    if False:
        while True:
            i = 10
    se.BooleanVar = BooleanVar
    se.StringVar = StringVar
    se.messagebox = messagebox

class Mock:

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

class GetTest(unittest.TestCase):

    def test_get(self):
        if False:
            print('Hello World!')
        saved_Engine = se.SearchEngine
        se.SearchEngine = Mock
        try:
            root = Mock()
            engine = se.get(root)
            self.assertIsInstance(engine, se.SearchEngine)
            self.assertIs(root._searchengine, engine)
            self.assertIs(se.get(root), engine)
        finally:
            se.SearchEngine = saved_Engine

class GetLineColTest(unittest.TestCase):

    def test_get_line_col(self):
        if False:
            return 10
        self.assertEqual(se.get_line_col('1.0'), (1, 0))
        self.assertEqual(se.get_line_col('1.11'), (1, 11))
        self.assertRaises(ValueError, se.get_line_col, '1.0 lineend')
        self.assertRaises(ValueError, se.get_line_col, 'end')

class GetSelectionTest(unittest.TestCase):

    def test_get_selection(self):
        if False:
            for i in range(10):
                print('nop')
        text = mockText()
        text.insert('1.0', 'Hello World!')

        def sel(s):
            if False:
                return 10
            if s == 'sel.first':
                return '1.0'
            if s == 'sel.last':
                return '1.12'
            raise TclError
        text.index = sel
        self.assertEqual(se.get_selection(text), ('1.0', '1.12'))

        def mark(s):
            if False:
                i = 10
                return i + 15
            if s == 'insert':
                return '1.5'
            raise TclError
        text.index = mark
        self.assertEqual(se.get_selection(text), ('1.5', '1.5'))

class ReverseSearchTest(unittest.TestCase):

    def test_search_reverse(self):
        if False:
            for i in range(10):
                print('nop')
        Equal = self.assertEqual
        line = "Here is an 'is' test text."
        prog = re.compile('is')
        Equal(se.search_reverse(prog, line, len(line)).span(), (12, 14))
        Equal(se.search_reverse(prog, line, 14).span(), (12, 14))
        Equal(se.search_reverse(prog, line, 13).span(), (5, 7))
        Equal(se.search_reverse(prog, line, 7).span(), (5, 7))
        Equal(se.search_reverse(prog, line, 6), None)

class SearchEngineTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.engine = se.SearchEngine(root=None)

    def test_is_get(self):
        if False:
            i = 10
            return i + 15
        engine = self.engine
        Equal = self.assertEqual
        Equal(engine.getpat(), '')
        engine.setpat('hello')
        Equal(engine.getpat(), 'hello')
        Equal(engine.isre(), False)
        engine.revar.set(1)
        Equal(engine.isre(), True)
        Equal(engine.iscase(), False)
        engine.casevar.set(1)
        Equal(engine.iscase(), True)
        Equal(engine.isword(), False)
        engine.wordvar.set(1)
        Equal(engine.isword(), True)
        Equal(engine.iswrap(), True)
        engine.wrapvar.set(0)
        Equal(engine.iswrap(), False)
        Equal(engine.isback(), False)
        engine.backvar.set(1)
        Equal(engine.isback(), True)

    def test_setcookedpat(self):
        if False:
            print('Hello World!')
        engine = self.engine
        engine.setcookedpat('\\s')
        self.assertEqual(engine.getpat(), '\\s')
        engine.revar.set(1)
        engine.setcookedpat('\\s')
        self.assertEqual(engine.getpat(), '\\\\s')

    def test_getcookedpat(self):
        if False:
            for i in range(10):
                print('nop')
        engine = self.engine
        Equal = self.assertEqual
        Equal(engine.getcookedpat(), '')
        engine.setpat('hello')
        Equal(engine.getcookedpat(), 'hello')
        engine.wordvar.set(True)
        Equal(engine.getcookedpat(), '\\bhello\\b')
        engine.wordvar.set(False)
        engine.setpat('\\s')
        Equal(engine.getcookedpat(), '\\\\s')
        engine.revar.set(True)
        Equal(engine.getcookedpat(), '\\s')

    def test_getprog(self):
        if False:
            while True:
                i = 10
        engine = self.engine
        Equal = self.assertEqual
        engine.setpat('Hello')
        temppat = engine.getprog()
        Equal(temppat.pattern, re.compile('Hello', re.IGNORECASE).pattern)
        engine.casevar.set(1)
        temppat = engine.getprog()
        Equal(temppat.pattern, re.compile('Hello').pattern, 0)
        engine.setpat('')
        Equal(engine.getprog(), None)
        Equal(Mbox.showerror.message, 'Error: Empty regular expression')
        engine.setpat('+')
        engine.revar.set(1)
        Equal(engine.getprog(), None)
        Equal(Mbox.showerror.message, 'Error: nothing to repeat\nPattern: +\nOffset: 0')

    def test_report_error(self):
        if False:
            return 10
        showerror = Mbox.showerror
        Equal = self.assertEqual
        pat = '[a-z'
        msg = 'unexpected end of regular expression'
        Equal(self.engine.report_error(pat, msg), None)
        Equal(showerror.title, 'Regular expression error')
        expected_message = 'Error: ' + msg + '\nPattern: [a-z'
        Equal(showerror.message, expected_message)
        Equal(self.engine.report_error(pat, msg, 5), None)
        Equal(showerror.title, 'Regular expression error')
        expected_message += '\nOffset: 5'
        Equal(showerror.message, expected_message)

class SearchTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.text = mockText()
        test_text = 'First line\nLine with target\nLast line\n'
        cls.text.insert('1.0', test_text)
        cls.pat = re.compile('target')
        cls.engine = se.SearchEngine(None)
        cls.engine.search_forward = lambda *args: ('f', args)
        cls.engine.search_backward = lambda *args: ('b', args)

    def test_search(self):
        if False:
            i = 10
            return i + 15
        Equal = self.assertEqual
        engine = self.engine
        search = engine.search_text
        text = self.text
        pat = self.pat
        engine.patvar.set(None)
        Equal(search(text), None)

        def mark(s):
            if False:
                while True:
                    i = 10
            if s == 'insert':
                return '1.5'
            raise TclError
        text.index = mark
        Equal(search(text, pat), ('f', (text, pat, 1, 5, True, False)))
        engine.wrapvar.set(False)
        Equal(search(text, pat), ('f', (text, pat, 1, 5, False, False)))
        engine.wrapvar.set(True)
        engine.backvar.set(True)
        Equal(search(text, pat), ('b', (text, pat, 1, 5, True, False)))
        engine.backvar.set(False)

        def sel(s):
            if False:
                print('Hello World!')
            if s == 'sel.first':
                return '2.10'
            if s == 'sel.last':
                return '2.16'
            raise TclError
        text.index = sel
        Equal(search(text, pat), ('f', (text, pat, 2, 16, True, False)))
        Equal(search(text, pat, True), ('f', (text, pat, 2, 10, True, True)))
        engine.backvar.set(True)
        Equal(search(text, pat), ('b', (text, pat, 2, 10, True, False)))
        Equal(search(text, pat, True), ('b', (text, pat, 2, 16, True, True)))

class ForwardBackwardTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.engine = se.SearchEngine(None)
        cls.text = mockText()
        cls.text.index = lambda index: '4.0'
        test_text = 'First line\nLine with target\nLast line\n'
        cls.text.insert('1.0', test_text)
        cls.pat = re.compile('target')
        cls.res = (2, (10, 16))
        cls.failpat = re.compile('xyz')
        cls.emptypat = re.compile('\\w*')

    def make_search(self, func):
        if False:
            for i in range(10):
                print('nop')

        def search(pat, line, col, wrap, ok=0):
            if False:
                print('Hello World!')
            res = func(self.text, pat, line, col, wrap, ok)
            return (res[0], res[1].span()) if res else res
        return search

    def test_search_forward(self):
        if False:
            return 10
        Equal = self.assertEqual
        forward = self.make_search(self.engine.search_forward)
        pat = self.pat
        Equal(forward(pat, 1, 0, True), self.res)
        Equal(forward(pat, 3, 0, True), self.res)
        Equal(forward(pat, 3, 0, False), None)
        Equal(forward(pat, 2, 10, False), self.res)
        Equal(forward(self.failpat, 1, 0, True), None)
        Equal(forward(self.emptypat, 2, 9, True, ok=True), (2, (9, 9)))
        Equal(forward(self.emptypat, 2, 10, True), self.res)

    def test_search_backward(self):
        if False:
            for i in range(10):
                print('nop')
        Equal = self.assertEqual
        backward = self.make_search(self.engine.search_backward)
        pat = self.pat
        Equal(backward(pat, 3, 5, True), self.res)
        Equal(backward(pat, 2, 0, True), self.res)
        Equal(backward(pat, 2, 0, False), None)
        Equal(backward(pat, 2, 16, False), self.res)
        Equal(backward(self.failpat, 3, 9, True), None)
        Equal(backward(self.emptypat, 2, 10, True, ok=True), (2, (9, 9)))
        Equal(backward(self.emptypat, 2, 9, True), (2, (5, 9)))
if __name__ == '__main__':
    unittest.main(verbosity=2)