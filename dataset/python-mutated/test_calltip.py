"""Test calltip, coverage 76%"""
from idlelib import calltip
import unittest
from unittest.mock import Mock
import textwrap
import types
import re
from idlelib.idle_test.mock_tk import Text

class TC:
    """doc"""
    tip = '(ai=None, *b)'

    def __init__(self, ai=None, *b):
        if False:
            print('Hello World!')
        'doc'
    __init__.tip = '(self, ai=None, *b)'

    def t1(self):
        if False:
            return 10
        'doc'
    t1.tip = '(self)'

    def t2(self, ai, b=None):
        if False:
            while True:
                i = 10
        'doc'
    t2.tip = '(self, ai, b=None)'

    def t3(self, ai, *args):
        if False:
            print('Hello World!')
        'doc'
    t3.tip = '(self, ai, *args)'

    def t4(self, *args):
        if False:
            print('Hello World!')
        'doc'
    t4.tip = '(self, *args)'

    def t5(self, ai, b=None, *args, **kw):
        if False:
            i = 10
            return i + 15
        'doc'
    t5.tip = '(self, ai, b=None, *args, **kw)'

    def t6(no, self):
        if False:
            print('Hello World!')
        'doc'
    t6.tip = '(no, self)'

    def __call__(self, ci):
        if False:
            for i in range(10):
                print('nop')
        'doc'
    __call__.tip = '(self, ci)'

    def nd(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @classmethod
    def cm(cls, a):
        if False:
            return 10
        'doc'

    @staticmethod
    def sm(b):
        if False:
            for i in range(10):
                print('nop')
        'doc'
tc = TC()
default_tip = calltip._default_callable_argspec
get_spec = calltip.get_argspec

class Get_argspecTest(unittest.TestCase):

    def test_builtins(self):
        if False:
            while True:
                i = 10

        def tiptest(obj, out):
            if False:
                return 10
            self.assertEqual(get_spec(obj), out)

        class List(list):
            """List() doc"""

        class SB:
            __call__ = None
        if List.__doc__ is not None:
            tiptest(List, f'(iterable=(), /)\n{List.__doc__}')
        tiptest(list.__new__, '(*args, **kwargs)\nCreate and return a new object.  See help(type) for accurate signature.')
        tiptest(list.__init__, '(self, /, *args, **kwargs)\nInitialize self.  See help(type(self)) for accurate signature.')
        append_doc = '\nAppend object to the end of the list.'
        tiptest(list.append, '(self, object, /)' + append_doc)
        tiptest(List.append, '(self, object, /)' + append_doc)
        tiptest([].append, '(object, /)' + append_doc)
        tiptest(types.MethodType, 'method(function, instance)')
        tiptest(SB(), default_tip)
        p = re.compile('')
        tiptest(re.sub, "(pattern, repl, string, count=0, flags=0)\nReturn the string obtained by replacing the leftmost\nnon-overlapping occurrences of the pattern in string by the\nreplacement repl.  repl can be either a string or a callable;\nif a string, backslash escapes in it are processed.  If it is\na callable, it's passed the Match object and must return")
        tiptest(p.sub, '(repl, string, count=0)\nReturn the string obtained by replacing the leftmost non-overlapping occurrences o...')

    def test_signature_wrap(self):
        if False:
            i = 10
            return i + 15
        if textwrap.TextWrapper.__doc__ is not None:
            self.assertEqual(get_spec(textwrap.TextWrapper), "(width=70, initial_indent='', subsequent_indent='', expand_tabs=True,\n    replace_whitespace=True, fix_sentence_endings=False, break_long_words=True,\n    drop_whitespace=True, break_on_hyphens=True, tabsize=8, *, max_lines=None,\n    placeholder=' [...]')\nObject for wrapping/filling text.  The public interface consists of\nthe wrap() and fill() methods; the other methods are just there for\nsubclasses to override in order to tweak the default behaviour.\nIf you want to completely replace the main wrapping algorithm,\nyou'll probably have to override _wrap_chunks().")

    def test_properly_formatted(self):
        if False:
            print('Hello World!')

        def foo(s='a' * 100):
            if False:
                print('Hello World!')
            pass

        def bar(s='a' * 100):
            if False:
                while True:
                    i = 10
            'Hello Guido'
            pass

        def baz(s='a' * 100, z='b' * 100):
            if False:
                while True:
                    i = 10
            pass
        indent = calltip._INDENT
        sfoo = "(s='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n" + indent + "aaaaaaaaaaaaaaaaaaa')"
        sbar = "(s='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n" + indent + "aaaaaaaaaaaaaaaaaaa')\nHello Guido"
        sbaz = "(s='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n" + indent + "aaaaaaaaaaaaaaaaaaa', z='bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n" + indent + "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')"
        for (func, doc) in [(foo, sfoo), (bar, sbar), (baz, sbaz)]:
            with self.subTest(func=func, doc=doc):
                self.assertEqual(get_spec(func), doc)

    def test_docline_truncation(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                return 10
            pass
        f.__doc__ = 'a' * 300
        self.assertEqual(get_spec(f), f"()\n{'a' * (calltip._MAX_COLS - 3) + '...'}")

    def test_multiline_docstring(self):
        if False:
            return 10
        self.assertEqual(get_spec(range), 'range(stop) -> range object\nrange(start, stop[, step]) -> range object')
        self.assertEqual(get_spec(bytes), 'bytes(iterable_of_ints) -> bytes\nbytes(string, encoding[, errors]) -> bytes\nbytes(bytes_or_buffer) -> immutable copy of bytes_or_buffer\nbytes(int) -> bytes object of size given by the parameter initialized with null bytes\nbytes() -> empty bytes object')

        def f():
            if False:
                print('Hello World!')
            pass
        f.__doc__ = 'a\n' * 15
        self.assertEqual(get_spec(f), '()' + '\na' * calltip._MAX_LINES)

    def test_functions(self):
        if False:
            return 10

        def t1():
            if False:
                for i in range(10):
                    print('nop')
            'doc'
        t1.tip = '()'

        def t2(a, b=None):
            if False:
                i = 10
                return i + 15
            'doc'
        t2.tip = '(a, b=None)'

        def t3(a, *args):
            if False:
                i = 10
                return i + 15
            'doc'
        t3.tip = '(a, *args)'

        def t4(*args):
            if False:
                return 10
            'doc'
        t4.tip = '(*args)'

        def t5(a, b=None, *args, **kw):
            if False:
                print('Hello World!')
            'doc'
        t5.tip = '(a, b=None, *args, **kw)'
        doc = '\ndoc' if t1.__doc__ is not None else ''
        for func in (t1, t2, t3, t4, t5, TC):
            with self.subTest(func=func):
                self.assertEqual(get_spec(func), func.tip + doc)

    def test_methods(self):
        if False:
            return 10
        doc = '\ndoc' if TC.__doc__ is not None else ''
        for meth in (TC.t1, TC.t2, TC.t3, TC.t4, TC.t5, TC.t6, TC.__call__):
            with self.subTest(meth=meth):
                self.assertEqual(get_spec(meth), meth.tip + doc)
        self.assertEqual(get_spec(TC.cm), '(a)' + doc)
        self.assertEqual(get_spec(TC.sm), '(b)' + doc)

    def test_bound_methods(self):
        if False:
            return 10
        doc = '\ndoc' if TC.__doc__ is not None else ''
        for (meth, mtip) in ((tc.t1, '()'), (tc.t4, '(*args)'), (tc.t6, '(self)'), (tc.__call__, '(ci)'), (tc, '(ci)'), (TC.cm, '(a)')):
            with self.subTest(meth=meth, mtip=mtip):
                self.assertEqual(get_spec(meth), mtip + doc)

    def test_starred_parameter(self):
        if False:
            i = 10
            return i + 15

        class C:

            def m1(*args):
                if False:
                    i = 10
                    return i + 15
                pass
        c = C()
        for (meth, mtip) in ((C.m1, '(*args)'), (c.m1, '(*args)')):
            with self.subTest(meth=meth, mtip=mtip):
                self.assertEqual(get_spec(meth), mtip)

    def test_invalid_method_get_spec(self):
        if False:
            while True:
                i = 10

        class C:

            def m2(**kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                pass

        class Test:

            def __call__(*, a):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        mtip = calltip._invalid_method
        self.assertEqual(get_spec(C().m2), mtip)
        self.assertEqual(get_spec(Test()), mtip)

    def test_non_ascii_name(self):
        if False:
            while True:
                i = 10
        uni = '(AΑАאاअḀあ, a)'
        assert calltip._first_param.sub('', uni) == '(a)'

    def test_no_docstring(self):
        if False:
            i = 10
            return i + 15
        for (meth, mtip) in ((TC.nd, '(self)'), (tc.nd, '()')):
            with self.subTest(meth=meth, mtip=mtip):
                self.assertEqual(get_spec(meth), mtip)

    def test_buggy_getattr_class(self):
        if False:
            return 10

        class NoCall:

            def __getattr__(self, name):
                if False:
                    i = 10
                    return i + 15
                raise IndexError

        class CallA(NoCall):

            def __call__(self, ci):
                if False:
                    for i in range(10):
                        print('nop')
                pass

        class CallB(NoCall):

            def __call__(oui, a, b, c):
                if False:
                    return 10
                pass
        for (meth, mtip) in ((NoCall, default_tip), (CallA, default_tip), (NoCall(), ''), (CallA(), '(ci)'), (CallB(), '(a, b, c)')):
            with self.subTest(meth=meth, mtip=mtip):
                self.assertEqual(get_spec(meth), mtip)

    def test_metaclass_class(self):
        if False:
            i = 10
            return i + 15

        class Type(type):
            __class__ = property({}.__getitem__, {}.__setitem__)

        class Object(metaclass=Type):
            __slots__ = '__class__'
        for (meth, mtip) in ((Type, get_spec(type)), (Object, default_tip), (Object(), '')):
            with self.subTest(meth=meth, mtip=mtip):
                self.assertEqual(get_spec(meth), mtip)

    def test_non_callables(self):
        if False:
            i = 10
            return i + 15
        for obj in (0, 0.0, '0', b'0', [], {}):
            with self.subTest(obj=obj):
                self.assertEqual(get_spec(obj), '')

class Get_entityTest(unittest.TestCase):

    def test_bad_entity(self):
        if False:
            while True:
                i = 10
        self.assertIsNone(calltip.get_entity('1/0'))

    def test_good_entity(self):
        if False:
            while True:
                i = 10
        self.assertIs(calltip.get_entity('int'), int)

class mock_Shell:
    """Return mock sufficient to pass to hyperparser."""

    def __init__(self, text):
        if False:
            return 10
        text.tag_prevrange = Mock(return_value=None)
        self.text = text
        self.prompt_last_line = '>>> '
        self.indentwidth = 4
        self.tabwidth = 8

class mock_TipWindow:

    def __init__(self):
        if False:
            return 10
        pass

    def showtip(self, text, parenleft, parenright):
        if False:
            for i in range(10):
                print('nop')
        self.args = (parenleft, parenright)
        (self.parenline, self.parencol) = map(int, parenleft.split('.'))

class WrappedCalltip(calltip.Calltip):

    def _make_tk_calltip_window(self):
        if False:
            i = 10
            return i + 15
        return mock_TipWindow()

    def remove_calltip_window(self, event=None):
        if False:
            while True:
                i = 10
        if self.active_calltip:
            self.active_calltip = None
            self.tips_removed += 1

    def fetch_tip(self, expression):
        if False:
            while True:
                i = 10
        return 'tip'

class CalltipTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.text = Text()
        cls.ct = WrappedCalltip(mock_Shell(cls.text))

    def setUp(self):
        if False:
            print('Hello World!')
        self.text.delete('1.0', 'end')
        self.ct.active_calltip = None
        self.ct.tips_removed = 0

    def open_close(self, testfunc):
        if False:
            return 10
        opentip = self.ct.open_calltip
        self.text.insert(1.0, 'f(')
        opentip(False)
        self.tip = self.ct.active_calltip
        testfunc(self)
        self.text.insert('insert', ')')
        opentip(False)
        self.assertIsNone(self.ct.active_calltip, None)

    def test_open_close(self):
        if False:
            return 10

        def args(self):
            if False:
                return 10
            self.assertEqual(self.tip.args, ('1.1', '1.end'))
        self.open_close(args)

    def test_repeated_force(self):
        if False:
            i = 10
            return i + 15

        def force(self):
            if False:
                while True:
                    i = 10
            for char in 'abc':
                self.text.insert('insert', 'a')
                self.ct.open_calltip(True)
                self.ct.open_calltip(True)
            self.assertIs(self.ct.active_calltip, self.tip)
        self.open_close(force)

    def test_repeated_parens(self):
        if False:
            i = 10
            return i + 15

        def parens(self):
            if False:
                print('Hello World!')
            for context in ('a', "'"):
                with self.subTest(context=context):
                    self.text.insert('insert', context)
                    for char in '(()())':
                        self.text.insert('insert', char)
                    self.assertIs(self.ct.active_calltip, self.tip)
            self.text.insert('insert', "'")
        self.open_close(parens)

    def test_comment_parens(self):
        if False:
            for i in range(10):
                print('nop')

        def comment(self):
            if False:
                return 10
            self.text.insert('insert', '# ')
            for char in '(()())':
                self.text.insert('insert', char)
            self.assertIs(self.ct.active_calltip, self.tip)
            self.text.insert('insert', '\n')
        self.open_close(comment)
if __name__ == '__main__':
    unittest.main(verbosity=2)