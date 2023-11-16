"""Tests for various magic functions specific to the terminal frontend."""
import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
MINIMAL_LAZY_MAGIC = '\nfrom IPython.core.magic import (\n    Magics,\n    magics_class,\n    line_magic,\n    cell_magic,\n)\n\n\n@magics_class\nclass LazyMagics(Magics):\n    @line_magic\n    def lazy_line(self, line):\n        print("Lazy Line")\n\n    @cell_magic\n    def lazy_cell(self, line, cell):\n        print("Lazy Cell")\n\n\ndef load_ipython_extension(ipython):\n    ipython.register_magics(LazyMagics)\n'

def check_cpaste(code, should_fail=False):
    if False:
        i = 10
        return i + 15
    "Execute code via 'cpaste' and ensure it was executed, unless\n    should_fail is set.\n    "
    ip.user_ns['code_ran'] = False
    src = StringIO()
    src.write(code)
    src.write('\n--\n')
    src.seek(0)
    stdin_save = sys.stdin
    sys.stdin = src
    try:
        context = tt.AssertPrints if should_fail else tt.AssertNotPrints
        with context('Traceback (most recent call last)'):
            ip.run_line_magic('cpaste', '')
        if not should_fail:
            assert ip.user_ns['code_ran'], '%r failed' % code
    finally:
        sys.stdin = stdin_save

def test_cpaste():
    if False:
        for i in range(10):
            print('nop')
    'Test cpaste magic'

    def runf():
        if False:
            return 10
        'Marker function: sets a flag when executed.\n        '
        ip.user_ns['code_ran'] = True
        return 'runf'
    tests = {'pass': ['runf()', 'In [1]: runf()', 'In [1]: if 1:\n   ...:     runf()', '> > > runf()', '>>> runf()', '   >>> runf()'], 'fail': ['1 + runf()', '++ runf()']}
    ip.user_ns['runf'] = runf
    for code in tests['pass']:
        check_cpaste(code)
    for code in tests['fail']:
        check_cpaste(code, should_fail=True)

class PasteTestCase(TestCase):
    """Multiple tests for clipboard pasting"""

    def paste(self, txt, flags='-q'):
        if False:
            i = 10
            return i + 15
        'Paste input text, by default in quiet mode'
        ip.hooks.clipboard_get = lambda : txt
        ip.run_line_magic('paste', flags)

    def setUp(self):
        if False:
            return 10
        self.original_clip = ip.hooks.clipboard_get

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        ip.hooks.clipboard_get = self.original_clip

    def test_paste(self):
        if False:
            while True:
                i = 10
        ip.user_ns.pop('x', None)
        self.paste('x = 1')
        self.assertEqual(ip.user_ns['x'], 1)
        ip.user_ns.pop('x')

    def test_paste_pyprompt(self):
        if False:
            for i in range(10):
                print('nop')
        ip.user_ns.pop('x', None)
        self.paste('>>> x=2')
        self.assertEqual(ip.user_ns['x'], 2)
        ip.user_ns.pop('x')

    def test_paste_py_multi(self):
        if False:
            while True:
                i = 10
        self.paste('\n        >>> x = [1,2,3]\n        >>> y = []\n        >>> for i in x:\n        ...     y.append(i**2)\n        ... \n        ')
        self.assertEqual(ip.user_ns['x'], [1, 2, 3])
        self.assertEqual(ip.user_ns['y'], [1, 4, 9])

    def test_paste_py_multi_r(self):
        if False:
            i = 10
            return i + 15
        'Now, test that self.paste -r works'
        self.test_paste_py_multi()
        self.assertEqual(ip.user_ns.pop('x'), [1, 2, 3])
        self.assertEqual(ip.user_ns.pop('y'), [1, 4, 9])
        self.assertFalse('x' in ip.user_ns)
        ip.run_line_magic('paste', '-r')
        self.assertEqual(ip.user_ns['x'], [1, 2, 3])
        self.assertEqual(ip.user_ns['y'], [1, 4, 9])

    def test_paste_email(self):
        if False:
            return 10
        'Test pasting of email-quoted contents'
        self.paste('        >> def foo(x):\n        >>     return x + 1\n        >> xx = foo(1.1)')
        self.assertEqual(ip.user_ns['xx'], 2.1)

    def test_paste_email2(self):
        if False:
            print('Hello World!')
        'Email again; some programs add a space also at each quoting level'
        self.paste('        > > def foo(x):\n        > >     return x + 1\n        > > yy = foo(2.1)     ')
        self.assertEqual(ip.user_ns['yy'], 3.1)

    def test_paste_email_py(self):
        if False:
            i = 10
            return i + 15
        'Email quoting of interactive input'
        self.paste('        >> >>> def f(x):\n        >> ...   return x+1\n        >> ... \n        >> >>> zz = f(2.5)      ')
        self.assertEqual(ip.user_ns['zz'], 3.5)

    def test_paste_echo(self):
        if False:
            i = 10
            return i + 15
        'Also test self.paste echoing, by temporarily faking the writer'
        w = StringIO()
        old_write = sys.stdout.write
        sys.stdout.write = w.write
        code = '\n        a = 100\n        b = 200'
        try:
            self.paste(code, '')
            out = w.getvalue()
        finally:
            sys.stdout.write = old_write
        self.assertEqual(ip.user_ns['a'], 100)
        self.assertEqual(ip.user_ns['b'], 200)
        assert out == code + '\n## -- End pasted text --\n'

    def test_paste_leading_commas(self):
        if False:
            i = 10
            return i + 15
        'Test multiline strings with leading commas'
        tm = ip.magics_manager.registry['TerminalMagics']
        s = 'a = """\n,1,2,3\n"""'
        ip.user_ns.pop('foo', None)
        tm.store_or_execute(s, 'foo')
        self.assertIn('foo', ip.user_ns)

    def test_paste_trailing_question(self):
        if False:
            while True:
                i = 10
        'Test pasting sources with trailing question marks'
        tm = ip.magics_manager.registry['TerminalMagics']
        s = "def funcfoo():\n   if True: #am i true?\n       return 'fooresult'\n"
        ip.user_ns.pop('funcfoo', None)
        self.paste(s)
        self.assertEqual(ip.user_ns['funcfoo'](), 'fooresult')