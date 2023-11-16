"""Define SearchEngine for search dialogs."""
import re
from tkinter import StringVar, BooleanVar, TclError
from tkinter import messagebox

def get(root):
    if False:
        for i in range(10):
            print('nop')
    'Return the singleton SearchEngine instance for the process.\n\n    The single SearchEngine saves settings between dialog instances.\n    If there is not a SearchEngine already, make one.\n    '
    if not hasattr(root, '_searchengine'):
        root._searchengine = SearchEngine(root)
    return root._searchengine

class SearchEngine:
    """Handles searching a text widget for Find, Replace, and Grep."""

    def __init__(self, root):
        if False:
            while True:
                i = 10
        'Initialize Variables that save search state.\n\n        The dialogs bind these to the UI elements present in the dialogs.\n        '
        self.root = root
        self.patvar = StringVar(root, '')
        self.revar = BooleanVar(root, False)
        self.casevar = BooleanVar(root, False)
        self.wordvar = BooleanVar(root, False)
        self.wrapvar = BooleanVar(root, True)
        self.backvar = BooleanVar(root, False)

    def getpat(self):
        if False:
            print('Hello World!')
        return self.patvar.get()

    def setpat(self, pat):
        if False:
            print('Hello World!')
        self.patvar.set(pat)

    def isre(self):
        if False:
            while True:
                i = 10
        return self.revar.get()

    def iscase(self):
        if False:
            i = 10
            return i + 15
        return self.casevar.get()

    def isword(self):
        if False:
            return 10
        return self.wordvar.get()

    def iswrap(self):
        if False:
            for i in range(10):
                print('nop')
        return self.wrapvar.get()

    def isback(self):
        if False:
            while True:
                i = 10
        return self.backvar.get()

    def setcookedpat(self, pat):
        if False:
            for i in range(10):
                print('nop')
        'Set pattern after escaping if re.'
        if self.isre():
            pat = re.escape(pat)
        self.setpat(pat)

    def getcookedpat(self):
        if False:
            print('Hello World!')
        pat = self.getpat()
        if not self.isre():
            pat = re.escape(pat)
        if self.isword():
            pat = '\\b%s\\b' % pat
        return pat

    def getprog(self):
        if False:
            for i in range(10):
                print('nop')
        'Return compiled cooked search pattern.'
        pat = self.getpat()
        if not pat:
            self.report_error(pat, 'Empty regular expression')
            return None
        pat = self.getcookedpat()
        flags = 0
        if not self.iscase():
            flags = flags | re.IGNORECASE
        try:
            prog = re.compile(pat, flags)
        except re.error as e:
            self.report_error(pat, e.msg, e.pos)
            return None
        return prog

    def report_error(self, pat, msg, col=None):
        if False:
            return 10
        msg = 'Error: ' + str(msg)
        if pat:
            msg = msg + '\nPattern: ' + str(pat)
        if col is not None:
            msg = msg + '\nOffset: ' + str(col)
        messagebox.showerror('Regular expression error', msg, master=self.root)

    def search_text(self, text, prog=None, ok=0):
        if False:
            return 10
        'Return (lineno, matchobj) or None for forward/backward search.\n\n        This function calls the right function with the right arguments.\n        It directly return the result of that call.\n\n        Text is a text widget. Prog is a precompiled pattern.\n        The ok parameter is a bit complicated as it has two effects.\n\n        If there is a selection, the search begin at either end,\n        depending on the direction setting and ok, with ok meaning that\n        the search starts with the selection. Otherwise, search begins\n        at the insert mark.\n\n        To aid progress, the search functions do not return an empty\n        match at the starting position unless ok is True.\n        '
        if not prog:
            prog = self.getprog()
            if not prog:
                return None
        wrap = self.wrapvar.get()
        (first, last) = get_selection(text)
        if self.isback():
            if ok:
                start = last
            else:
                start = first
            (line, col) = get_line_col(start)
            res = self.search_backward(text, prog, line, col, wrap, ok)
        else:
            if ok:
                start = first
            else:
                start = last
            (line, col) = get_line_col(start)
            res = self.search_forward(text, prog, line, col, wrap, ok)
        return res

    def search_forward(self, text, prog, line, col, wrap, ok=0):
        if False:
            return 10
        wrapped = 0
        startline = line
        chars = text.get('%d.0' % line, '%d.0' % (line + 1))
        while chars:
            m = prog.search(chars[:-1], col)
            if m:
                if ok or m.end() > col:
                    return (line, m)
            line = line + 1
            if wrapped and line > startline:
                break
            col = 0
            ok = 1
            chars = text.get('%d.0' % line, '%d.0' % (line + 1))
            if not chars and wrap:
                wrapped = 1
                wrap = 0
                line = 1
                chars = text.get('1.0', '2.0')
        return None

    def search_backward(self, text, prog, line, col, wrap, ok=0):
        if False:
            i = 10
            return i + 15
        wrapped = 0
        startline = line
        chars = text.get('%d.0' % line, '%d.0' % (line + 1))
        while 1:
            m = search_reverse(prog, chars[:-1], col)
            if m:
                if ok or m.start() < col:
                    return (line, m)
            line = line - 1
            if wrapped and line < startline:
                break
            ok = 1
            if line <= 0:
                if not wrap:
                    break
                wrapped = 1
                wrap = 0
                pos = text.index('end-1c')
                (line, col) = map(int, pos.split('.'))
            chars = text.get('%d.0' % line, '%d.0' % (line + 1))
            col = len(chars) - 1
        return None

def search_reverse(prog, chars, col):
    if False:
        while True:
            i = 10
    'Search backwards and return an re match object or None.\n\n    This is done by searching forwards until there is no match.\n    Prog: compiled re object with a search method returning a match.\n    Chars: line of text, without \\n.\n    Col: stop index for the search; the limit for match.end().\n    '
    m = prog.search(chars)
    if not m:
        return None
    found = None
    (i, j) = m.span()
    while i < col and j <= col:
        found = m
        if i == j:
            j = j + 1
        m = prog.search(chars, j)
        if not m:
            break
        (i, j) = m.span()
    return found

def get_selection(text):
    if False:
        i = 10
        return i + 15
    "Return tuple of 'line.col' indexes from selection or insert mark.\n    "
    try:
        first = text.index('sel.first')
        last = text.index('sel.last')
    except TclError:
        first = last = None
    if not first:
        first = text.index('insert')
    if not last:
        last = first
    return (first, last)

def get_line_col(index):
    if False:
        while True:
            i = 10
    "Return (line, col) tuple of ints from 'line.col' string."
    (line, col) = map(int, index.split('.'))
    return (line, col)
if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_searchengine', verbosity=2)