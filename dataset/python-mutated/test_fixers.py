""" Test suite for the fixer modules """
import os
from itertools import chain
from operator import itemgetter
from lib2to3 import pygram, fixer_util
from lib2to3.tests import support

class FixerTestCase(support.TestCase):

    def setUp(self, fix_list=None, fixer_pkg='lib2to3', options=None):
        if False:
            return 10
        if fix_list is None:
            fix_list = [self.fixer]
        self.refactor = support.get_refactorer(fixer_pkg, fix_list, options)
        self.fixer_log = []
        self.filename = '<string>'
        for fixer in chain(self.refactor.pre_order, self.refactor.post_order):
            fixer.log = self.fixer_log

    def _check(self, before, after):
        if False:
            for i in range(10):
                print('nop')
        before = support.reformat(before)
        after = support.reformat(after)
        tree = self.refactor.refactor_string(before, self.filename)
        self.assertEqual(after, str(tree))
        return tree

    def check(self, before, after, ignore_warnings=False):
        if False:
            while True:
                i = 10
        tree = self._check(before, after)
        self.assertTrue(tree.was_changed)
        if not ignore_warnings:
            self.assertEqual(self.fixer_log, [])

    def warns(self, before, after, message, unchanged=False):
        if False:
            print('Hello World!')
        tree = self._check(before, after)
        self.assertIn(message, ''.join(self.fixer_log))
        if not unchanged:
            self.assertTrue(tree.was_changed)

    def warns_unchanged(self, before, message):
        if False:
            i = 10
            return i + 15
        self.warns(before, before, message, unchanged=True)

    def unchanged(self, before, ignore_warnings=False):
        if False:
            while True:
                i = 10
        self._check(before, before)
        if not ignore_warnings:
            self.assertEqual(self.fixer_log, [])

    def assert_runs_after(self, *names):
        if False:
            i = 10
            return i + 15
        fixes = [self.fixer]
        fixes.extend(names)
        r = support.get_refactorer('lib2to3', fixes)
        (pre, post) = r.get_fixers()
        n = 'fix_' + self.fixer
        if post and post[-1].__class__.__module__.endswith(n):
            return
        if pre and pre[-1].__class__.__module__.endswith(n) and (not post):
            return
        self.fail('Fixer run order (%s) is incorrect; %s should be last.' % (', '.join([x.__class__.__module__ for x in pre + post]), n))

class Test_ne(FixerTestCase):
    fixer = 'ne'

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'if x <> y:\n            pass'
        a = 'if x != y:\n            pass'
        self.check(b, a)

    def test_no_spaces(self):
        if False:
            print('Hello World!')
        b = 'if x<>y:\n            pass'
        a = 'if x!=y:\n            pass'
        self.check(b, a)

    def test_chained(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'if x<>y<>z:\n            pass'
        a = 'if x!=y!=z:\n            pass'
        self.check(b, a)

class Test_has_key(FixerTestCase):
    fixer = 'has_key'

    def test_1(self):
        if False:
            return 10
        b = 'x = d.has_key("x") or d.has_key("y")'
        a = 'x = "x" in d or "y" in d'
        self.check(b, a)

    def test_2(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'x = a.b.c.d.has_key("x") ** 3'
        a = 'x = ("x" in a.b.c.d) ** 3'
        self.check(b, a)

    def test_3(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'x = a.b.has_key(1 + 2).__repr__()'
        a = 'x = (1 + 2 in a.b).__repr__()'
        self.check(b, a)

    def test_4(self):
        if False:
            i = 10
            return i + 15
        b = 'x = a.b.has_key(1 + 2).__repr__() ** -3 ** 4'
        a = 'x = (1 + 2 in a.b).__repr__() ** -3 ** 4'
        self.check(b, a)

    def test_5(self):
        if False:
            return 10
        b = 'x = a.has_key(f or g)'
        a = 'x = (f or g) in a'
        self.check(b, a)

    def test_6(self):
        if False:
            while True:
                i = 10
        b = 'x = a + b.has_key(c)'
        a = 'x = a + (c in b)'
        self.check(b, a)

    def test_7(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'x = a.has_key(lambda: 12)'
        a = 'x = (lambda: 12) in a'
        self.check(b, a)

    def test_8(self):
        if False:
            print('Hello World!')
        b = 'x = a.has_key(a for a in b)'
        a = 'x = (a for a in b) in a'
        self.check(b, a)

    def test_9(self):
        if False:
            while True:
                i = 10
        b = 'if not a.has_key(b): pass'
        a = 'if b not in a: pass'
        self.check(b, a)

    def test_10(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'if not a.has_key(b).__repr__(): pass'
        a = 'if not (b in a).__repr__(): pass'
        self.check(b, a)

    def test_11(self):
        if False:
            return 10
        b = 'if not a.has_key(b) ** 2: pass'
        a = 'if not (b in a) ** 2: pass'
        self.check(b, a)

class Test_apply(FixerTestCase):
    fixer = 'apply'

    def test_1(self):
        if False:
            i = 10
            return i + 15
        b = 'x = apply(f, g + h)'
        a = 'x = f(*g + h)'
        self.check(b, a)

    def test_2(self):
        if False:
            while True:
                i = 10
        b = 'y = apply(f, g, h)'
        a = 'y = f(*g, **h)'
        self.check(b, a)

    def test_3(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'z = apply(fs[0], g or h, h or g)'
        a = 'z = fs[0](*g or h, **h or g)'
        self.check(b, a)

    def test_4(self):
        if False:
            i = 10
            return i + 15
        b = 'apply(f, (x, y) + t)'
        a = 'f(*(x, y) + t)'
        self.check(b, a)

    def test_5(self):
        if False:
            i = 10
            return i + 15
        b = 'apply(f, args,)'
        a = 'f(*args)'
        self.check(b, a)

    def test_6(self):
        if False:
            return 10
        b = 'apply(f, args, kwds,)'
        a = 'f(*args, **kwds)'
        self.check(b, a)

    def test_complex_1(self):
        if False:
            print('Hello World!')
        b = 'x = apply(f+g, args)'
        a = 'x = (f+g)(*args)'
        self.check(b, a)

    def test_complex_2(self):
        if False:
            print('Hello World!')
        b = 'x = apply(f*g, args)'
        a = 'x = (f*g)(*args)'
        self.check(b, a)

    def test_complex_3(self):
        if False:
            i = 10
            return i + 15
        b = 'x = apply(f**g, args)'
        a = 'x = (f**g)(*args)'
        self.check(b, a)

    def test_dotted_name(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'x = apply(f.g, args)'
        a = 'x = f.g(*args)'
        self.check(b, a)

    def test_subscript(self):
        if False:
            while True:
                i = 10
        b = 'x = apply(f[x], args)'
        a = 'x = f[x](*args)'
        self.check(b, a)

    def test_call(self):
        if False:
            while True:
                i = 10
        b = 'x = apply(f(), args)'
        a = 'x = f()(*args)'
        self.check(b, a)

    def test_extreme(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'x = apply(a.b.c.d.e.f, args, kwds)'
        a = 'x = a.b.c.d.e.f(*args, **kwds)'
        self.check(b, a)

    def test_weird_comments(self):
        if False:
            while True:
                i = 10
        b = 'apply(   # foo\n          f, # bar\n          args)'
        a = 'f(*args)'
        self.check(b, a)

    def test_unchanged_1(self):
        if False:
            for i in range(10):
                print('nop')
        s = 'apply()'
        self.unchanged(s)

    def test_unchanged_2(self):
        if False:
            return 10
        s = 'apply(f)'
        self.unchanged(s)

    def test_unchanged_3(self):
        if False:
            for i in range(10):
                print('nop')
        s = 'apply(f,)'
        self.unchanged(s)

    def test_unchanged_4(self):
        if False:
            print('Hello World!')
        s = 'apply(f, args, kwds, extras)'
        self.unchanged(s)

    def test_unchanged_5(self):
        if False:
            return 10
        s = 'apply(f, *args, **kwds)'
        self.unchanged(s)

    def test_unchanged_6(self):
        if False:
            for i in range(10):
                print('nop')
        s = 'apply(f, *args)'
        self.unchanged(s)

    def test_unchanged_6b(self):
        if False:
            print('Hello World!')
        s = 'apply(f, **kwds)'
        self.unchanged(s)

    def test_unchanged_7(self):
        if False:
            return 10
        s = 'apply(func=f, args=args, kwds=kwds)'
        self.unchanged(s)

    def test_unchanged_8(self):
        if False:
            print('Hello World!')
        s = 'apply(f, args=args, kwds=kwds)'
        self.unchanged(s)

    def test_unchanged_9(self):
        if False:
            print('Hello World!')
        s = 'apply(f, args, kwds=kwds)'
        self.unchanged(s)

    def test_space_1(self):
        if False:
            i = 10
            return i + 15
        a = 'apply(  f,  args,   kwds)'
        b = 'f(*args, **kwds)'
        self.check(a, b)

    def test_space_2(self):
        if False:
            print('Hello World!')
        a = 'apply(  f  ,args,kwds   )'
        b = 'f(*args, **kwds)'
        self.check(a, b)

class Test_reload(FixerTestCase):
    fixer = 'reload'

    def test(self):
        if False:
            i = 10
            return i + 15
        b = 'reload(a)'
        a = 'import importlib\nimportlib.reload(a)'
        self.check(b, a)

    def test_comment(self):
        if False:
            print('Hello World!')
        b = 'reload( a ) # comment'
        a = 'import importlib\nimportlib.reload( a ) # comment'
        self.check(b, a)
        b = 'reload( a )  # comment'
        a = 'import importlib\nimportlib.reload( a )  # comment'
        self.check(b, a)

    def test_space(self):
        if False:
            while True:
                i = 10
        b = 'reload( a )'
        a = 'import importlib\nimportlib.reload( a )'
        self.check(b, a)
        b = 'reload( a)'
        a = 'import importlib\nimportlib.reload( a)'
        self.check(b, a)
        b = 'reload(a )'
        a = 'import importlib\nimportlib.reload(a )'
        self.check(b, a)

    def test_unchanged(self):
        if False:
            print('Hello World!')
        s = 'reload(a=1)'
        self.unchanged(s)
        s = 'reload(f, g)'
        self.unchanged(s)
        s = 'reload(f, *h)'
        self.unchanged(s)
        s = 'reload(f, *h, **i)'
        self.unchanged(s)
        s = 'reload(f, **i)'
        self.unchanged(s)
        s = 'reload(*h, **i)'
        self.unchanged(s)
        s = 'reload(*h)'
        self.unchanged(s)
        s = 'reload(**i)'
        self.unchanged(s)
        s = 'reload()'
        self.unchanged(s)

class Test_intern(FixerTestCase):
    fixer = 'intern'

    def test_prefix_preservation(self):
        if False:
            return 10
        b = 'x =   intern(  a  )'
        a = 'import sys\nx =   sys.intern(  a  )'
        self.check(b, a)
        b = 'y = intern("b" # test\n              )'
        a = 'import sys\ny = sys.intern("b" # test\n              )'
        self.check(b, a)
        b = 'z = intern(a+b+c.d,   )'
        a = 'import sys\nz = sys.intern(a+b+c.d,   )'
        self.check(b, a)

    def test(self):
        if False:
            while True:
                i = 10
        b = 'x = intern(a)'
        a = 'import sys\nx = sys.intern(a)'
        self.check(b, a)
        b = 'z = intern(a+b+c.d,)'
        a = 'import sys\nz = sys.intern(a+b+c.d,)'
        self.check(b, a)
        b = 'intern("y%s" % 5).replace("y", "")'
        a = 'import sys\nsys.intern("y%s" % 5).replace("y", "")'
        self.check(b, a)

    def test_unchanged(self):
        if False:
            while True:
                i = 10
        s = 'intern(a=1)'
        self.unchanged(s)
        s = 'intern(f, g)'
        self.unchanged(s)
        s = 'intern(*h)'
        self.unchanged(s)
        s = 'intern(**i)'
        self.unchanged(s)
        s = 'intern()'
        self.unchanged(s)

class Test_reduce(FixerTestCase):
    fixer = 'reduce'

    def test_simple_call(self):
        if False:
            return 10
        b = 'reduce(a, b, c)'
        a = 'from functools import reduce\nreduce(a, b, c)'
        self.check(b, a)

    def test_bug_7253(self):
        if False:
            return 10
        b = 'def x(arg): reduce(sum, [])'
        a = 'from functools import reduce\ndef x(arg): reduce(sum, [])'
        self.check(b, a)

    def test_call_with_lambda(self):
        if False:
            print('Hello World!')
        b = 'reduce(lambda x, y: x + y, seq)'
        a = 'from functools import reduce\nreduce(lambda x, y: x + y, seq)'
        self.check(b, a)

    def test_unchanged(self):
        if False:
            print('Hello World!')
        s = 'reduce(a)'
        self.unchanged(s)
        s = 'reduce(a, b=42)'
        self.unchanged(s)
        s = 'reduce(a, b, c, d)'
        self.unchanged(s)
        s = 'reduce(**c)'
        self.unchanged(s)
        s = 'reduce()'
        self.unchanged(s)

class Test_print(FixerTestCase):
    fixer = 'print'

    def test_prefix_preservation(self):
        if False:
            while True:
                i = 10
        b = 'print 1,   1+1,   1+1+1'
        a = 'print(1,   1+1,   1+1+1)'
        self.check(b, a)

    def test_idempotency(self):
        if False:
            i = 10
            return i + 15
        s = 'print()'
        self.unchanged(s)
        s = "print('')"
        self.unchanged(s)

    def test_idempotency_print_as_function(self):
        if False:
            for i in range(10):
                print('nop')
        self.refactor.driver.grammar = pygram.python_grammar_no_print_statement
        s = 'print(1, 1+1, 1+1+1)'
        self.unchanged(s)
        s = 'print()'
        self.unchanged(s)
        s = "print('')"
        self.unchanged(s)

    def test_1(self):
        if False:
            while True:
                i = 10
        b = 'print 1, 1+1, 1+1+1'
        a = 'print(1, 1+1, 1+1+1)'
        self.check(b, a)

    def test_2(self):
        if False:
            print('Hello World!')
        b = 'print 1, 2'
        a = 'print(1, 2)'
        self.check(b, a)

    def test_3(self):
        if False:
            i = 10
            return i + 15
        b = 'print'
        a = 'print()'
        self.check(b, a)

    def test_4(self):
        if False:
            while True:
                i = 10
        b = 'print whatever; print'
        a = 'print(whatever); print()'
        self.check(b, a)

    def test_5(self):
        if False:
            i = 10
            return i + 15
        b = 'print; print whatever;'
        a = 'print(); print(whatever);'
        self.check(b, a)

    def test_tuple(self):
        if False:
            i = 10
            return i + 15
        b = 'print (a, b, c)'
        a = 'print((a, b, c))'
        self.check(b, a)

    def test_trailing_comma_1(self):
        if False:
            return 10
        b = 'print 1, 2, 3,'
        a = "print(1, 2, 3, end=' ')"
        self.check(b, a)

    def test_trailing_comma_2(self):
        if False:
            return 10
        b = 'print 1, 2,'
        a = "print(1, 2, end=' ')"
        self.check(b, a)

    def test_trailing_comma_3(self):
        if False:
            print('Hello World!')
        b = 'print 1,'
        a = "print(1, end=' ')"
        self.check(b, a)

    def test_vargs_without_trailing_comma(self):
        if False:
            i = 10
            return i + 15
        b = 'print >>sys.stderr, 1, 2, 3'
        a = 'print(1, 2, 3, file=sys.stderr)'
        self.check(b, a)

    def test_with_trailing_comma(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'print >>sys.stderr, 1, 2,'
        a = "print(1, 2, end=' ', file=sys.stderr)"
        self.check(b, a)

    def test_no_trailing_comma(self):
        if False:
            print('Hello World!')
        b = 'print >>sys.stderr, 1+1'
        a = 'print(1+1, file=sys.stderr)'
        self.check(b, a)

    def test_spaces_before_file(self):
        if False:
            return 10
        b = 'print >>  sys.stderr'
        a = 'print(file=sys.stderr)'
        self.check(b, a)

    def test_with_future_print_function(self):
        if False:
            while True:
                i = 10
        s = "from __future__ import print_function\nprint('Hai!', end=' ')"
        self.unchanged(s)
        b = "print 'Hello, world!'"
        a = "print('Hello, world!')"
        self.check(b, a)

class Test_exec(FixerTestCase):
    fixer = 'exec'

    def test_prefix_preservation(self):
        if False:
            while True:
                i = 10
        b = '  exec code in ns1,   ns2'
        a = '  exec(code, ns1,   ns2)'
        self.check(b, a)

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'exec code'
        a = 'exec(code)'
        self.check(b, a)

    def test_with_globals(self):
        if False:
            print('Hello World!')
        b = 'exec code in ns'
        a = 'exec(code, ns)'
        self.check(b, a)

    def test_with_globals_locals(self):
        if False:
            i = 10
            return i + 15
        b = 'exec code in ns1, ns2'
        a = 'exec(code, ns1, ns2)'
        self.check(b, a)

    def test_complex_1(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'exec (a.b()) in ns'
        a = 'exec((a.b()), ns)'
        self.check(b, a)

    def test_complex_2(self):
        if False:
            while True:
                i = 10
        b = 'exec a.b() + c in ns'
        a = 'exec(a.b() + c, ns)'
        self.check(b, a)

    def test_unchanged_1(self):
        if False:
            for i in range(10):
                print('nop')
        s = 'exec(code)'
        self.unchanged(s)

    def test_unchanged_2(self):
        if False:
            return 10
        s = 'exec (code)'
        self.unchanged(s)

    def test_unchanged_3(self):
        if False:
            for i in range(10):
                print('nop')
        s = 'exec(code, ns)'
        self.unchanged(s)

    def test_unchanged_4(self):
        if False:
            while True:
                i = 10
        s = 'exec(code, ns1, ns2)'
        self.unchanged(s)

class Test_repr(FixerTestCase):
    fixer = 'repr'

    def test_prefix_preservation(self):
        if False:
            return 10
        b = 'x =   `1 + 2`'
        a = 'x =   repr(1 + 2)'
        self.check(b, a)

    def test_simple_1(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'x = `1 + 2`'
        a = 'x = repr(1 + 2)'
        self.check(b, a)

    def test_simple_2(self):
        if False:
            print('Hello World!')
        b = 'y = `x`'
        a = 'y = repr(x)'
        self.check(b, a)

    def test_complex(self):
        if False:
            i = 10
            return i + 15
        b = 'z = `y`.__repr__()'
        a = 'z = repr(y).__repr__()'
        self.check(b, a)

    def test_tuple(self):
        if False:
            i = 10
            return i + 15
        b = 'x = `1, 2, 3`'
        a = 'x = repr((1, 2, 3))'
        self.check(b, a)

    def test_nested(self):
        if False:
            return 10
        b = 'x = `1 + `2``'
        a = 'x = repr(1 + repr(2))'
        self.check(b, a)

    def test_nested_tuples(self):
        if False:
            i = 10
            return i + 15
        b = 'x = `1, 2 + `3, 4``'
        a = 'x = repr((1, 2 + repr((3, 4))))'
        self.check(b, a)

class Test_except(FixerTestCase):
    fixer = 'except'

    def test_prefix_preservation(self):
        if False:
            print('Hello World!')
        b = '\n            try:\n                pass\n            except (RuntimeError, ImportError),    e:\n                pass'
        a = '\n            try:\n                pass\n            except (RuntimeError, ImportError) as    e:\n                pass'
        self.check(b, a)

    def test_simple(self):
        if False:
            print('Hello World!')
        b = '\n            try:\n                pass\n            except Foo, e:\n                pass'
        a = '\n            try:\n                pass\n            except Foo as e:\n                pass'
        self.check(b, a)

    def test_simple_no_space_before_target(self):
        if False:
            for i in range(10):
                print('nop')
        b = '\n            try:\n                pass\n            except Foo,e:\n                pass'
        a = '\n            try:\n                pass\n            except Foo as e:\n                pass'
        self.check(b, a)

    def test_tuple_unpack(self):
        if False:
            for i in range(10):
                print('nop')
        b = '\n            def foo():\n                try:\n                    pass\n                except Exception, (f, e):\n                    pass\n                except ImportError, e:\n                    pass'
        a = '\n            def foo():\n                try:\n                    pass\n                except Exception as xxx_todo_changeme:\n                    (f, e) = xxx_todo_changeme.args\n                    pass\n                except ImportError as e:\n                    pass'
        self.check(b, a)

    def test_multi_class(self):
        if False:
            return 10
        b = '\n            try:\n                pass\n            except (RuntimeError, ImportError), e:\n                pass'
        a = '\n            try:\n                pass\n            except (RuntimeError, ImportError) as e:\n                pass'
        self.check(b, a)

    def test_list_unpack(self):
        if False:
            for i in range(10):
                print('nop')
        b = '\n            try:\n                pass\n            except Exception, [a, b]:\n                pass'
        a = '\n            try:\n                pass\n            except Exception as xxx_todo_changeme:\n                [a, b] = xxx_todo_changeme.args\n                pass'
        self.check(b, a)

    def test_weird_target_1(self):
        if False:
            return 10
        b = '\n            try:\n                pass\n            except Exception, d[5]:\n                pass'
        a = '\n            try:\n                pass\n            except Exception as xxx_todo_changeme:\n                d[5] = xxx_todo_changeme\n                pass'
        self.check(b, a)

    def test_weird_target_2(self):
        if False:
            print('Hello World!')
        b = '\n            try:\n                pass\n            except Exception, a.foo:\n                pass'
        a = '\n            try:\n                pass\n            except Exception as xxx_todo_changeme:\n                a.foo = xxx_todo_changeme\n                pass'
        self.check(b, a)

    def test_weird_target_3(self):
        if False:
            while True:
                i = 10
        b = '\n            try:\n                pass\n            except Exception, a().foo:\n                pass'
        a = '\n            try:\n                pass\n            except Exception as xxx_todo_changeme:\n                a().foo = xxx_todo_changeme\n                pass'
        self.check(b, a)

    def test_bare_except(self):
        if False:
            i = 10
            return i + 15
        b = '\n            try:\n                pass\n            except Exception, a:\n                pass\n            except:\n                pass'
        a = '\n            try:\n                pass\n            except Exception as a:\n                pass\n            except:\n                pass'
        self.check(b, a)

    def test_bare_except_and_else_finally(self):
        if False:
            for i in range(10):
                print('nop')
        b = '\n            try:\n                pass\n            except Exception, a:\n                pass\n            except:\n                pass\n            else:\n                pass\n            finally:\n                pass'
        a = '\n            try:\n                pass\n            except Exception as a:\n                pass\n            except:\n                pass\n            else:\n                pass\n            finally:\n                pass'
        self.check(b, a)

    def test_multi_fixed_excepts_before_bare_except(self):
        if False:
            for i in range(10):
                print('nop')
        b = '\n            try:\n                pass\n            except TypeError, b:\n                pass\n            except Exception, a:\n                pass\n            except:\n                pass'
        a = '\n            try:\n                pass\n            except TypeError as b:\n                pass\n            except Exception as a:\n                pass\n            except:\n                pass'
        self.check(b, a)

    def test_one_line_suites(self):
        if False:
            for i in range(10):
                print('nop')
        b = '\n            try: raise TypeError\n            except TypeError, e:\n                pass\n            '
        a = '\n            try: raise TypeError\n            except TypeError as e:\n                pass\n            '
        self.check(b, a)
        b = '\n            try:\n                raise TypeError\n            except TypeError, e: pass\n            '
        a = '\n            try:\n                raise TypeError\n            except TypeError as e: pass\n            '
        self.check(b, a)
        b = '\n            try: raise TypeError\n            except TypeError, e: pass\n            '
        a = '\n            try: raise TypeError\n            except TypeError as e: pass\n            '
        self.check(b, a)
        b = '\n            try: raise TypeError\n            except TypeError, e: pass\n            else: function()\n            finally: done()\n            '
        a = '\n            try: raise TypeError\n            except TypeError as e: pass\n            else: function()\n            finally: done()\n            '
        self.check(b, a)

    def test_unchanged_1(self):
        if False:
            while True:
                i = 10
        s = '\n            try:\n                pass\n            except:\n                pass'
        self.unchanged(s)

    def test_unchanged_2(self):
        if False:
            i = 10
            return i + 15
        s = '\n            try:\n                pass\n            except Exception:\n                pass'
        self.unchanged(s)

    def test_unchanged_3(self):
        if False:
            i = 10
            return i + 15
        s = '\n            try:\n                pass\n            except (Exception, SystemExit):\n                pass'
        self.unchanged(s)

class Test_raise(FixerTestCase):
    fixer = 'raise'

    def test_basic(self):
        if False:
            return 10
        b = 'raise Exception, 5'
        a = 'raise Exception(5)'
        self.check(b, a)

    def test_prefix_preservation(self):
        if False:
            print('Hello World!')
        b = 'raise Exception,5'
        a = 'raise Exception(5)'
        self.check(b, a)
        b = 'raise   Exception,    5'
        a = 'raise   Exception(5)'
        self.check(b, a)

    def test_with_comments(self):
        if False:
            return 10
        b = 'raise Exception, 5 # foo'
        a = 'raise Exception(5) # foo'
        self.check(b, a)
        b = 'raise E, (5, 6) % (a, b) # foo'
        a = 'raise E((5, 6) % (a, b)) # foo'
        self.check(b, a)
        b = 'def foo():\n                    raise Exception, 5, 6 # foo'
        a = 'def foo():\n                    raise Exception(5).with_traceback(6) # foo'
        self.check(b, a)

    def test_None_value(self):
        if False:
            while True:
                i = 10
        b = 'raise Exception(5), None, tb'
        a = 'raise Exception(5).with_traceback(tb)'
        self.check(b, a)

    def test_tuple_value(self):
        if False:
            while True:
                i = 10
        b = 'raise Exception, (5, 6, 7)'
        a = 'raise Exception(5, 6, 7)'
        self.check(b, a)

    def test_tuple_detection(self):
        if False:
            i = 10
            return i + 15
        b = 'raise E, (5, 6) % (a, b)'
        a = 'raise E((5, 6) % (a, b))'
        self.check(b, a)

    def test_tuple_exc_1(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'raise (((E1, E2), E3), E4), V'
        a = 'raise E1(V)'
        self.check(b, a)

    def test_tuple_exc_2(self):
        if False:
            return 10
        b = 'raise (E1, (E2, E3), E4), V'
        a = 'raise E1(V)'
        self.check(b, a)

    def test_string_exc(self):
        if False:
            return 10
        s = "raise 'foo'"
        self.warns_unchanged(s, 'Python 3 does not support string exceptions')

    def test_string_exc_val(self):
        if False:
            i = 10
            return i + 15
        s = 'raise "foo", 5'
        self.warns_unchanged(s, 'Python 3 does not support string exceptions')

    def test_string_exc_val_tb(self):
        if False:
            while True:
                i = 10
        s = 'raise "foo", 5, 6'
        self.warns_unchanged(s, 'Python 3 does not support string exceptions')

    def test_tb_1(self):
        if False:
            while True:
                i = 10
        b = 'def foo():\n                    raise Exception, 5, 6'
        a = 'def foo():\n                    raise Exception(5).with_traceback(6)'
        self.check(b, a)

    def test_tb_2(self):
        if False:
            print('Hello World!')
        b = 'def foo():\n                    a = 5\n                    raise Exception, 5, 6\n                    b = 6'
        a = 'def foo():\n                    a = 5\n                    raise Exception(5).with_traceback(6)\n                    b = 6'
        self.check(b, a)

    def test_tb_3(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'def foo():\n                    raise Exception,5,6'
        a = 'def foo():\n                    raise Exception(5).with_traceback(6)'
        self.check(b, a)

    def test_tb_4(self):
        if False:
            i = 10
            return i + 15
        b = 'def foo():\n                    a = 5\n                    raise Exception,5,6\n                    b = 6'
        a = 'def foo():\n                    a = 5\n                    raise Exception(5).with_traceback(6)\n                    b = 6'
        self.check(b, a)

    def test_tb_5(self):
        if False:
            i = 10
            return i + 15
        b = 'def foo():\n                    raise Exception, (5, 6, 7), 6'
        a = 'def foo():\n                    raise Exception(5, 6, 7).with_traceback(6)'
        self.check(b, a)

    def test_tb_6(self):
        if False:
            print('Hello World!')
        b = 'def foo():\n                    a = 5\n                    raise Exception, (5, 6, 7), 6\n                    b = 6'
        a = 'def foo():\n                    a = 5\n                    raise Exception(5, 6, 7).with_traceback(6)\n                    b = 6'
        self.check(b, a)

class Test_throw(FixerTestCase):
    fixer = 'throw'

    def test_1(self):
        if False:
            print('Hello World!')
        b = 'g.throw(Exception, 5)'
        a = 'g.throw(Exception(5))'
        self.check(b, a)

    def test_2(self):
        if False:
            while True:
                i = 10
        b = 'g.throw(Exception,5)'
        a = 'g.throw(Exception(5))'
        self.check(b, a)

    def test_3(self):
        if False:
            print('Hello World!')
        b = 'g.throw(Exception, (5, 6, 7))'
        a = 'g.throw(Exception(5, 6, 7))'
        self.check(b, a)

    def test_4(self):
        if False:
            print('Hello World!')
        b = '5 + g.throw(Exception, 5)'
        a = '5 + g.throw(Exception(5))'
        self.check(b, a)

    def test_warn_1(self):
        if False:
            i = 10
            return i + 15
        s = 'g.throw("foo")'
        self.warns_unchanged(s, 'Python 3 does not support string exceptions')

    def test_warn_2(self):
        if False:
            return 10
        s = 'g.throw("foo", 5)'
        self.warns_unchanged(s, 'Python 3 does not support string exceptions')

    def test_warn_3(self):
        if False:
            return 10
        s = 'g.throw("foo", 5, 6)'
        self.warns_unchanged(s, 'Python 3 does not support string exceptions')

    def test_untouched_1(self):
        if False:
            print('Hello World!')
        s = 'g.throw(Exception)'
        self.unchanged(s)

    def test_untouched_2(self):
        if False:
            i = 10
            return i + 15
        s = 'g.throw(Exception(5, 6))'
        self.unchanged(s)

    def test_untouched_3(self):
        if False:
            return 10
        s = '5 + g.throw(Exception(5, 6))'
        self.unchanged(s)

    def test_tb_1(self):
        if False:
            print('Hello World!')
        b = 'def foo():\n                    g.throw(Exception, 5, 6)'
        a = 'def foo():\n                    g.throw(Exception(5).with_traceback(6))'
        self.check(b, a)

    def test_tb_2(self):
        if False:
            print('Hello World!')
        b = 'def foo():\n                    a = 5\n                    g.throw(Exception, 5, 6)\n                    b = 6'
        a = 'def foo():\n                    a = 5\n                    g.throw(Exception(5).with_traceback(6))\n                    b = 6'
        self.check(b, a)

    def test_tb_3(self):
        if False:
            i = 10
            return i + 15
        b = 'def foo():\n                    g.throw(Exception,5,6)'
        a = 'def foo():\n                    g.throw(Exception(5).with_traceback(6))'
        self.check(b, a)

    def test_tb_4(self):
        if False:
            i = 10
            return i + 15
        b = 'def foo():\n                    a = 5\n                    g.throw(Exception,5,6)\n                    b = 6'
        a = 'def foo():\n                    a = 5\n                    g.throw(Exception(5).with_traceback(6))\n                    b = 6'
        self.check(b, a)

    def test_tb_5(self):
        if False:
            i = 10
            return i + 15
        b = 'def foo():\n                    g.throw(Exception, (5, 6, 7), 6)'
        a = 'def foo():\n                    g.throw(Exception(5, 6, 7).with_traceback(6))'
        self.check(b, a)

    def test_tb_6(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'def foo():\n                    a = 5\n                    g.throw(Exception, (5, 6, 7), 6)\n                    b = 6'
        a = 'def foo():\n                    a = 5\n                    g.throw(Exception(5, 6, 7).with_traceback(6))\n                    b = 6'
        self.check(b, a)

    def test_tb_7(self):
        if False:
            i = 10
            return i + 15
        b = 'def foo():\n                    a + g.throw(Exception, 5, 6)'
        a = 'def foo():\n                    a + g.throw(Exception(5).with_traceback(6))'
        self.check(b, a)

    def test_tb_8(self):
        if False:
            i = 10
            return i + 15
        b = 'def foo():\n                    a = 5\n                    a + g.throw(Exception, 5, 6)\n                    b = 6'
        a = 'def foo():\n                    a = 5\n                    a + g.throw(Exception(5).with_traceback(6))\n                    b = 6'
        self.check(b, a)

class Test_long(FixerTestCase):
    fixer = 'long'

    def test_1(self):
        if False:
            print('Hello World!')
        b = 'x = long(x)'
        a = 'x = int(x)'
        self.check(b, a)

    def test_2(self):
        if False:
            print('Hello World!')
        b = 'y = isinstance(x, long)'
        a = 'y = isinstance(x, int)'
        self.check(b, a)

    def test_3(self):
        if False:
            i = 10
            return i + 15
        b = 'z = type(x) in (int, long)'
        a = 'z = type(x) in (int, int)'
        self.check(b, a)

    def test_unchanged(self):
        if False:
            return 10
        s = 'long = True'
        self.unchanged(s)
        s = 's.long = True'
        self.unchanged(s)
        s = 'def long(): pass'
        self.unchanged(s)
        s = 'class long(): pass'
        self.unchanged(s)
        s = 'def f(long): pass'
        self.unchanged(s)
        s = 'def f(g, long): pass'
        self.unchanged(s)
        s = 'def f(x, long=True): pass'
        self.unchanged(s)

    def test_prefix_preservation(self):
        if False:
            return 10
        b = 'x =   long(  x  )'
        a = 'x =   int(  x  )'
        self.check(b, a)

class Test_execfile(FixerTestCase):
    fixer = 'execfile'

    def test_conversion(self):
        if False:
            while True:
                i = 10
        b = 'execfile("fn")'
        a = 'exec(compile(open("fn", "rb").read(), "fn", \'exec\'))'
        self.check(b, a)
        b = 'execfile("fn", glob)'
        a = 'exec(compile(open("fn", "rb").read(), "fn", \'exec\'), glob)'
        self.check(b, a)
        b = 'execfile("fn", glob, loc)'
        a = 'exec(compile(open("fn", "rb").read(), "fn", \'exec\'), glob, loc)'
        self.check(b, a)
        b = 'execfile("fn", globals=glob)'
        a = 'exec(compile(open("fn", "rb").read(), "fn", \'exec\'), globals=glob)'
        self.check(b, a)
        b = 'execfile("fn", locals=loc)'
        a = 'exec(compile(open("fn", "rb").read(), "fn", \'exec\'), locals=loc)'
        self.check(b, a)
        b = 'execfile("fn", globals=glob, locals=loc)'
        a = 'exec(compile(open("fn", "rb").read(), "fn", \'exec\'), globals=glob, locals=loc)'
        self.check(b, a)

    def test_spacing(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'execfile( "fn" )'
        a = 'exec(compile(open( "fn", "rb" ).read(), "fn", \'exec\'))'
        self.check(b, a)
        b = 'execfile("fn",  globals = glob)'
        a = 'exec(compile(open("fn", "rb").read(), "fn", \'exec\'),  globals = glob)'
        self.check(b, a)

class Test_isinstance(FixerTestCase):
    fixer = 'isinstance'

    def test_remove_multiple_items(self):
        if False:
            i = 10
            return i + 15
        b = 'isinstance(x, (int, int, int))'
        a = 'isinstance(x, int)'
        self.check(b, a)
        b = 'isinstance(x, (int, float, int, int, float))'
        a = 'isinstance(x, (int, float))'
        self.check(b, a)
        b = 'isinstance(x, (int, float, int, int, float, str))'
        a = 'isinstance(x, (int, float, str))'
        self.check(b, a)
        b = 'isinstance(foo() + bar(), (x(), y(), x(), int, int))'
        a = 'isinstance(foo() + bar(), (x(), y(), x(), int))'
        self.check(b, a)

    def test_prefix_preservation(self):
        if False:
            i = 10
            return i + 15
        b = 'if    isinstance(  foo(), (  bar, bar, baz )) : pass'
        a = 'if    isinstance(  foo(), (  bar, baz )) : pass'
        self.check(b, a)

    def test_unchanged(self):
        if False:
            i = 10
            return i + 15
        self.unchanged('isinstance(x, (str, int))')

class Test_dict(FixerTestCase):
    fixer = 'dict'

    def test_prefix_preservation(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'if   d. keys  (  )  : pass'
        a = 'if   list(d. keys  (  ))  : pass'
        self.check(b, a)
        b = 'if   d. items  (  )  : pass'
        a = 'if   list(d. items  (  ))  : pass'
        self.check(b, a)
        b = 'if   d. iterkeys  ( )  : pass'
        a = 'if   iter(d. keys  ( ))  : pass'
        self.check(b, a)
        b = '[i for i in    d.  iterkeys(  )  ]'
        a = '[i for i in    d.  keys(  )  ]'
        self.check(b, a)
        b = 'if   d. viewkeys  ( )  : pass'
        a = 'if   d. keys  ( )  : pass'
        self.check(b, a)
        b = '[i for i in    d.  viewkeys(  )  ]'
        a = '[i for i in    d.  keys(  )  ]'
        self.check(b, a)

    def test_trailing_comment(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'd.keys() # foo'
        a = 'list(d.keys()) # foo'
        self.check(b, a)
        b = 'd.items()  # foo'
        a = 'list(d.items())  # foo'
        self.check(b, a)
        b = 'd.iterkeys()  # foo'
        a = 'iter(d.keys())  # foo'
        self.check(b, a)
        b = '[i for i in d.iterkeys() # foo\n               ]'
        a = '[i for i in d.keys() # foo\n               ]'
        self.check(b, a)
        b = '[i for i in d.iterkeys() # foo\n               ]'
        a = '[i for i in d.keys() # foo\n               ]'
        self.check(b, a)
        b = 'd.viewitems()  # foo'
        a = 'd.items()  # foo'
        self.check(b, a)

    def test_unchanged(self):
        if False:
            return 10
        for wrapper in fixer_util.consuming_calls:
            s = 's = %s(d.keys())' % wrapper
            self.unchanged(s)
            s = 's = %s(d.values())' % wrapper
            self.unchanged(s)
            s = 's = %s(d.items())' % wrapper
            self.unchanged(s)

    def test_01(self):
        if False:
            return 10
        b = 'd.keys()'
        a = 'list(d.keys())'
        self.check(b, a)
        b = 'a[0].foo().keys()'
        a = 'list(a[0].foo().keys())'
        self.check(b, a)

    def test_02(self):
        if False:
            while True:
                i = 10
        b = 'd.items()'
        a = 'list(d.items())'
        self.check(b, a)

    def test_03(self):
        if False:
            return 10
        b = 'd.values()'
        a = 'list(d.values())'
        self.check(b, a)

    def test_04(self):
        if False:
            return 10
        b = 'd.iterkeys()'
        a = 'iter(d.keys())'
        self.check(b, a)

    def test_05(self):
        if False:
            print('Hello World!')
        b = 'd.iteritems()'
        a = 'iter(d.items())'
        self.check(b, a)

    def test_06(self):
        if False:
            print('Hello World!')
        b = 'd.itervalues()'
        a = 'iter(d.values())'
        self.check(b, a)

    def test_07(self):
        if False:
            i = 10
            return i + 15
        s = 'list(d.keys())'
        self.unchanged(s)

    def test_08(self):
        if False:
            i = 10
            return i + 15
        s = 'sorted(d.keys())'
        self.unchanged(s)

    def test_09(self):
        if False:
            return 10
        b = 'iter(d.keys())'
        a = 'iter(list(d.keys()))'
        self.check(b, a)

    def test_10(self):
        if False:
            while True:
                i = 10
        b = 'foo(d.keys())'
        a = 'foo(list(d.keys()))'
        self.check(b, a)

    def test_11(self):
        if False:
            i = 10
            return i + 15
        b = 'for i in d.keys(): print i'
        a = 'for i in list(d.keys()): print i'
        self.check(b, a)

    def test_12(self):
        if False:
            return 10
        b = 'for i in d.iterkeys(): print i'
        a = 'for i in d.keys(): print i'
        self.check(b, a)

    def test_13(self):
        if False:
            print('Hello World!')
        b = '[i for i in d.keys()]'
        a = '[i for i in list(d.keys())]'
        self.check(b, a)

    def test_14(self):
        if False:
            while True:
                i = 10
        b = '[i for i in d.iterkeys()]'
        a = '[i for i in d.keys()]'
        self.check(b, a)

    def test_15(self):
        if False:
            while True:
                i = 10
        b = '(i for i in d.keys())'
        a = '(i for i in list(d.keys()))'
        self.check(b, a)

    def test_16(self):
        if False:
            print('Hello World!')
        b = '(i for i in d.iterkeys())'
        a = '(i for i in d.keys())'
        self.check(b, a)

    def test_17(self):
        if False:
            while True:
                i = 10
        b = 'iter(d.iterkeys())'
        a = 'iter(d.keys())'
        self.check(b, a)

    def test_18(self):
        if False:
            return 10
        b = 'list(d.iterkeys())'
        a = 'list(d.keys())'
        self.check(b, a)

    def test_19(self):
        if False:
            i = 10
            return i + 15
        b = 'sorted(d.iterkeys())'
        a = 'sorted(d.keys())'
        self.check(b, a)

    def test_20(self):
        if False:
            return 10
        b = 'foo(d.iterkeys())'
        a = 'foo(iter(d.keys()))'
        self.check(b, a)

    def test_21(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'print h.iterkeys().next()'
        a = 'print iter(h.keys()).next()'
        self.check(b, a)

    def test_22(self):
        if False:
            while True:
                i = 10
        b = 'print h.keys()[0]'
        a = 'print list(h.keys())[0]'
        self.check(b, a)

    def test_23(self):
        if False:
            print('Hello World!')
        b = 'print list(h.iterkeys().next())'
        a = 'print list(iter(h.keys()).next())'
        self.check(b, a)

    def test_24(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'for x in h.keys()[0]: print x'
        a = 'for x in list(h.keys())[0]: print x'
        self.check(b, a)

    def test_25(self):
        if False:
            while True:
                i = 10
        b = 'd.viewkeys()'
        a = 'd.keys()'
        self.check(b, a)

    def test_26(self):
        if False:
            print('Hello World!')
        b = 'd.viewitems()'
        a = 'd.items()'
        self.check(b, a)

    def test_27(self):
        if False:
            i = 10
            return i + 15
        b = 'd.viewvalues()'
        a = 'd.values()'
        self.check(b, a)

    def test_28(self):
        if False:
            print('Hello World!')
        b = '[i for i in d.viewkeys()]'
        a = '[i for i in d.keys()]'
        self.check(b, a)

    def test_29(self):
        if False:
            for i in range(10):
                print('nop')
        b = '(i for i in d.viewkeys())'
        a = '(i for i in d.keys())'
        self.check(b, a)

    def test_30(self):
        if False:
            i = 10
            return i + 15
        b = 'iter(d.viewkeys())'
        a = 'iter(d.keys())'
        self.check(b, a)

    def test_31(self):
        if False:
            print('Hello World!')
        b = 'list(d.viewkeys())'
        a = 'list(d.keys())'
        self.check(b, a)

    def test_32(self):
        if False:
            while True:
                i = 10
        b = 'sorted(d.viewkeys())'
        a = 'sorted(d.keys())'
        self.check(b, a)

class Test_xrange(FixerTestCase):
    fixer = 'xrange'

    def test_prefix_preservation(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'x =    xrange(  10  )'
        a = 'x =    range(  10  )'
        self.check(b, a)
        b = 'x = xrange(  1  ,  10   )'
        a = 'x = range(  1  ,  10   )'
        self.check(b, a)
        b = 'x = xrange(  0  ,  10 ,  2 )'
        a = 'x = range(  0  ,  10 ,  2 )'
        self.check(b, a)

    def test_single_arg(self):
        if False:
            return 10
        b = 'x = xrange(10)'
        a = 'x = range(10)'
        self.check(b, a)

    def test_two_args(self):
        if False:
            i = 10
            return i + 15
        b = 'x = xrange(1, 10)'
        a = 'x = range(1, 10)'
        self.check(b, a)

    def test_three_args(self):
        if False:
            return 10
        b = 'x = xrange(0, 10, 2)'
        a = 'x = range(0, 10, 2)'
        self.check(b, a)

    def test_wrap_in_list(self):
        if False:
            print('Hello World!')
        b = 'x = range(10, 3, 9)'
        a = 'x = list(range(10, 3, 9))'
        self.check(b, a)
        b = 'x = foo(range(10, 3, 9))'
        a = 'x = foo(list(range(10, 3, 9)))'
        self.check(b, a)
        b = 'x = range(10, 3, 9) + [4]'
        a = 'x = list(range(10, 3, 9)) + [4]'
        self.check(b, a)
        b = 'x = range(10)[::-1]'
        a = 'x = list(range(10))[::-1]'
        self.check(b, a)
        b = 'x = range(10)  [3]'
        a = 'x = list(range(10))  [3]'
        self.check(b, a)

    def test_xrange_in_for(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'for i in xrange(10):\n    j=i'
        a = 'for i in range(10):\n    j=i'
        self.check(b, a)
        b = '[i for i in xrange(10)]'
        a = '[i for i in range(10)]'
        self.check(b, a)

    def test_range_in_for(self):
        if False:
            i = 10
            return i + 15
        self.unchanged('for i in range(10): pass')
        self.unchanged('[i for i in range(10)]')

    def test_in_contains_test(self):
        if False:
            for i in range(10):
                print('nop')
        self.unchanged('x in range(10, 3, 9)')

    def test_in_consuming_context(self):
        if False:
            print('Hello World!')
        for call in fixer_util.consuming_calls:
            self.unchanged('a = %s(range(10))' % call)

class Test_xrange_with_reduce(FixerTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(Test_xrange_with_reduce, self).setUp(['xrange', 'reduce'])

    def test_double_transform(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'reduce(x, xrange(5))'
        a = 'from functools import reduce\nreduce(x, range(5))'
        self.check(b, a)

class Test_raw_input(FixerTestCase):
    fixer = 'raw_input'

    def test_prefix_preservation(self):
        if False:
            print('Hello World!')
        b = 'x =    raw_input(   )'
        a = 'x =    input(   )'
        self.check(b, a)
        b = "x = raw_input(   ''   )"
        a = "x = input(   ''   )"
        self.check(b, a)

    def test_1(self):
        if False:
            i = 10
            return i + 15
        b = 'x = raw_input()'
        a = 'x = input()'
        self.check(b, a)

    def test_2(self):
        if False:
            print('Hello World!')
        b = "x = raw_input('')"
        a = "x = input('')"
        self.check(b, a)

    def test_3(self):
        if False:
            return 10
        b = "x = raw_input('prompt')"
        a = "x = input('prompt')"
        self.check(b, a)

    def test_4(self):
        if False:
            i = 10
            return i + 15
        b = 'x = raw_input(foo(a) + 6)'
        a = 'x = input(foo(a) + 6)'
        self.check(b, a)

    def test_5(self):
        if False:
            print('Hello World!')
        b = 'x = raw_input(invite).split()'
        a = 'x = input(invite).split()'
        self.check(b, a)

    def test_6(self):
        if False:
            return 10
        b = 'x = raw_input(invite) . split ()'
        a = 'x = input(invite) . split ()'
        self.check(b, a)

    def test_8(self):
        if False:
            print('Hello World!')
        b = 'x = int(raw_input())'
        a = 'x = int(input())'
        self.check(b, a)

class Test_funcattrs(FixerTestCase):
    fixer = 'funcattrs'
    attrs = ['closure', 'doc', 'name', 'defaults', 'code', 'globals', 'dict']

    def test(self):
        if False:
            print('Hello World!')
        for attr in self.attrs:
            b = 'a.func_%s' % attr
            a = 'a.__%s__' % attr
            self.check(b, a)
            b = 'self.foo.func_%s.foo_bar' % attr
            a = 'self.foo.__%s__.foo_bar' % attr
            self.check(b, a)

    def test_unchanged(self):
        if False:
            while True:
                i = 10
        for attr in self.attrs:
            s = 'foo(func_%s + 5)' % attr
            self.unchanged(s)
            s = 'f(foo.__%s__)' % attr
            self.unchanged(s)
            s = 'f(foo.__%s__.foo)' % attr
            self.unchanged(s)

class Test_xreadlines(FixerTestCase):
    fixer = 'xreadlines'

    def test_call(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'for x in f.xreadlines(): pass'
        a = 'for x in f: pass'
        self.check(b, a)
        b = 'for x in foo().xreadlines(): pass'
        a = 'for x in foo(): pass'
        self.check(b, a)
        b = 'for x in (5 + foo()).xreadlines(): pass'
        a = 'for x in (5 + foo()): pass'
        self.check(b, a)

    def test_attr_ref(self):
        if False:
            return 10
        b = 'foo(f.xreadlines + 5)'
        a = 'foo(f.__iter__ + 5)'
        self.check(b, a)
        b = 'foo(f().xreadlines + 5)'
        a = 'foo(f().__iter__ + 5)'
        self.check(b, a)
        b = 'foo((5 + f()).xreadlines + 5)'
        a = 'foo((5 + f()).__iter__ + 5)'
        self.check(b, a)

    def test_unchanged(self):
        if False:
            i = 10
            return i + 15
        s = 'for x in f.xreadlines(5): pass'
        self.unchanged(s)
        s = 'for x in f.xreadlines(k=5): pass'
        self.unchanged(s)
        s = 'for x in f.xreadlines(*k, **v): pass'
        self.unchanged(s)
        s = 'foo(xreadlines)'
        self.unchanged(s)

class ImportsFixerTests:

    def test_import_module(self):
        if False:
            i = 10
            return i + 15
        for (old, new) in self.modules.items():
            b = 'import %s' % old
            a = 'import %s' % new
            self.check(b, a)
            b = 'import foo, %s, bar' % old
            a = 'import foo, %s, bar' % new
            self.check(b, a)

    def test_import_from(self):
        if False:
            for i in range(10):
                print('nop')
        for (old, new) in self.modules.items():
            b = 'from %s import foo' % old
            a = 'from %s import foo' % new
            self.check(b, a)
            b = 'from %s import foo, bar' % old
            a = 'from %s import foo, bar' % new
            self.check(b, a)
            b = 'from %s import (yes, no)' % old
            a = 'from %s import (yes, no)' % new
            self.check(b, a)

    def test_import_module_as(self):
        if False:
            print('Hello World!')
        for (old, new) in self.modules.items():
            b = 'import %s as foo_bar' % old
            a = 'import %s as foo_bar' % new
            self.check(b, a)
            b = 'import %s as foo_bar' % old
            a = 'import %s as foo_bar' % new
            self.check(b, a)

    def test_import_from_as(self):
        if False:
            while True:
                i = 10
        for (old, new) in self.modules.items():
            b = 'from %s import foo as bar' % old
            a = 'from %s import foo as bar' % new
            self.check(b, a)

    def test_star(self):
        if False:
            return 10
        for (old, new) in self.modules.items():
            b = 'from %s import *' % old
            a = 'from %s import *' % new
            self.check(b, a)

    def test_import_module_usage(self):
        if False:
            print('Hello World!')
        for (old, new) in self.modules.items():
            b = '\n                import %s\n                foo(%s.bar)\n                ' % (old, old)
            a = '\n                import %s\n                foo(%s.bar)\n                ' % (new, new)
            self.check(b, a)
            b = '\n                from %s import x\n                %s = 23\n                ' % (old, old)
            a = '\n                from %s import x\n                %s = 23\n                ' % (new, old)
            self.check(b, a)
            s = '\n                def f():\n                    %s.method()\n                ' % (old,)
            self.unchanged(s)
            b = '\n                import %s\n                %s.bar(%s.foo)\n                ' % (old, old, old)
            a = '\n                import %s\n                %s.bar(%s.foo)\n                ' % (new, new, new)
            self.check(b, a)
            b = '\n                import %s\n                x.%s\n                ' % (old, old)
            a = '\n                import %s\n                x.%s\n                ' % (new, old)
            self.check(b, a)

class Test_imports(FixerTestCase, ImportsFixerTests):
    fixer = 'imports'
    from ..fixes.fix_imports import MAPPING as modules

    def test_multiple_imports(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'import urlparse, cStringIO'
        a = 'import urllib.parse, io'
        self.check(b, a)

    def test_multiple_imports_as(self):
        if False:
            while True:
                i = 10
        b = '\n            import copy_reg as bar, HTMLParser as foo, urlparse\n            s = urlparse.spam(bar.foo())\n            '
        a = '\n            import copyreg as bar, html.parser as foo, urllib.parse\n            s = urllib.parse.spam(bar.foo())\n            '
        self.check(b, a)

class Test_imports2(FixerTestCase, ImportsFixerTests):
    fixer = 'imports2'
    from ..fixes.fix_imports2 import MAPPING as modules

class Test_imports_fixer_order(FixerTestCase, ImportsFixerTests):

    def setUp(self):
        if False:
            return 10
        super(Test_imports_fixer_order, self).setUp(['imports', 'imports2'])
        from ..fixes.fix_imports2 import MAPPING as mapping2
        self.modules = mapping2.copy()
        from ..fixes.fix_imports import MAPPING as mapping1
        for key in ('dbhash', 'dumbdbm', 'dbm', 'gdbm'):
            self.modules[key] = mapping1[key]

    def test_after_local_imports_refactoring(self):
        if False:
            while True:
                i = 10
        for fix in ('imports', 'imports2'):
            self.fixer = fix
            self.assert_runs_after('import')

class Test_urllib(FixerTestCase):
    fixer = 'urllib'
    from ..fixes.fix_urllib import MAPPING as modules

    def test_import_module(self):
        if False:
            i = 10
            return i + 15
        for (old, changes) in self.modules.items():
            b = 'import %s' % old
            a = 'import %s' % ', '.join(map(itemgetter(0), changes))
            self.check(b, a)

    def test_import_from(self):
        if False:
            for i in range(10):
                print('nop')
        for (old, changes) in self.modules.items():
            all_members = []
            for (new, members) in changes:
                for member in members:
                    all_members.append(member)
                    b = 'from %s import %s' % (old, member)
                    a = 'from %s import %s' % (new, member)
                    self.check(b, a)
                    s = 'from foo import %s' % member
                    self.unchanged(s)
                b = 'from %s import %s' % (old, ', '.join(members))
                a = 'from %s import %s' % (new, ', '.join(members))
                self.check(b, a)
                s = 'from foo import %s' % ', '.join(members)
                self.unchanged(s)
            b = 'from %s import %s' % (old, ', '.join(all_members))
            a = '\n'.join(['from %s import %s' % (new, ', '.join(members)) for (new, members) in changes])
            self.check(b, a)

    def test_import_module_as(self):
        if False:
            i = 10
            return i + 15
        for old in self.modules:
            s = 'import %s as foo' % old
            self.warns_unchanged(s, 'This module is now multiple modules')

    def test_import_from_as(self):
        if False:
            i = 10
            return i + 15
        for (old, changes) in self.modules.items():
            for (new, members) in changes:
                for member in members:
                    b = 'from %s import %s as foo_bar' % (old, member)
                    a = 'from %s import %s as foo_bar' % (new, member)
                    self.check(b, a)
                    b = 'from %s import %s as blah, %s' % (old, member, member)
                    a = 'from %s import %s as blah, %s' % (new, member, member)
                    self.check(b, a)

    def test_star(self):
        if False:
            return 10
        for old in self.modules:
            s = 'from %s import *' % old
            self.warns_unchanged(s, 'Cannot handle star imports')

    def test_indented(self):
        if False:
            i = 10
            return i + 15
        b = '\ndef foo():\n    from urllib import urlencode, urlopen\n'
        a = '\ndef foo():\n    from urllib.parse import urlencode\n    from urllib.request import urlopen\n'
        self.check(b, a)
        b = '\ndef foo():\n    other()\n    from urllib import urlencode, urlopen\n'
        a = '\ndef foo():\n    other()\n    from urllib.parse import urlencode\n    from urllib.request import urlopen\n'
        self.check(b, a)

    def test_single_import(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'from urllib import getproxies'
        a = 'from urllib.request import getproxies'
        self.check(b, a)

    def test_import_module_usage(self):
        if False:
            i = 10
            return i + 15
        for (old, changes) in self.modules.items():
            for (new, members) in changes:
                for member in members:
                    new_import = ', '.join([n for (n, mems) in self.modules[old]])
                    b = '\n                        import %s\n                        foo(%s.%s)\n                        ' % (old, old, member)
                    a = '\n                        import %s\n                        foo(%s.%s)\n                        ' % (new_import, new, member)
                    self.check(b, a)
                    b = '\n                        import %s\n                        %s.%s(%s.%s)\n                        ' % (old, old, member, old, member)
                    a = '\n                        import %s\n                        %s.%s(%s.%s)\n                        ' % (new_import, new, member, new, member)
                    self.check(b, a)

class Test_input(FixerTestCase):
    fixer = 'input'

    def test_prefix_preservation(self):
        if False:
            i = 10
            return i + 15
        b = 'x =   input(   )'
        a = 'x =   eval(input(   ))'
        self.check(b, a)
        b = "x = input(   ''   )"
        a = "x = eval(input(   ''   ))"
        self.check(b, a)

    def test_trailing_comment(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'x = input()  #  foo'
        a = 'x = eval(input())  #  foo'
        self.check(b, a)

    def test_idempotency(self):
        if False:
            for i in range(10):
                print('nop')
        s = 'x = eval(input())'
        self.unchanged(s)
        s = "x = eval(input(''))"
        self.unchanged(s)
        s = 'x = eval(input(foo(5) + 9))'
        self.unchanged(s)

    def test_1(self):
        if False:
            return 10
        b = 'x = input()'
        a = 'x = eval(input())'
        self.check(b, a)

    def test_2(self):
        if False:
            return 10
        b = "x = input('')"
        a = "x = eval(input(''))"
        self.check(b, a)

    def test_3(self):
        if False:
            print('Hello World!')
        b = "x = input('prompt')"
        a = "x = eval(input('prompt'))"
        self.check(b, a)

    def test_4(self):
        if False:
            while True:
                i = 10
        b = 'x = input(foo(5) + 9)'
        a = 'x = eval(input(foo(5) + 9))'
        self.check(b, a)

class Test_tuple_params(FixerTestCase):
    fixer = 'tuple_params'

    def test_unchanged_1(self):
        if False:
            i = 10
            return i + 15
        s = 'def foo(): pass'
        self.unchanged(s)

    def test_unchanged_2(self):
        if False:
            return 10
        s = 'def foo(a, b, c): pass'
        self.unchanged(s)

    def test_unchanged_3(self):
        if False:
            print('Hello World!')
        s = 'def foo(a=3, b=4, c=5): pass'
        self.unchanged(s)

    def test_1(self):
        if False:
            while True:
                i = 10
        b = '\n            def foo(((a, b), c)):\n                x = 5'
        a = '\n            def foo(xxx_todo_changeme):\n                ((a, b), c) = xxx_todo_changeme\n                x = 5'
        self.check(b, a)

    def test_2(self):
        if False:
            for i in range(10):
                print('nop')
        b = '\n            def foo(((a, b), c), d):\n                x = 5'
        a = '\n            def foo(xxx_todo_changeme, d):\n                ((a, b), c) = xxx_todo_changeme\n                x = 5'
        self.check(b, a)

    def test_3(self):
        if False:
            while True:
                i = 10
        b = '\n            def foo(((a, b), c), d) -> e:\n                x = 5'
        a = '\n            def foo(xxx_todo_changeme, d) -> e:\n                ((a, b), c) = xxx_todo_changeme\n                x = 5'
        self.check(b, a)

    def test_semicolon(self):
        if False:
            return 10
        b = '\n            def foo(((a, b), c)): x = 5; y = 7'
        a = '\n            def foo(xxx_todo_changeme): ((a, b), c) = xxx_todo_changeme; x = 5; y = 7'
        self.check(b, a)

    def test_keywords(self):
        if False:
            for i in range(10):
                print('nop')
        b = '\n            def foo(((a, b), c), d, e=5) -> z:\n                x = 5'
        a = '\n            def foo(xxx_todo_changeme, d, e=5) -> z:\n                ((a, b), c) = xxx_todo_changeme\n                x = 5'
        self.check(b, a)

    def test_varargs(self):
        if False:
            while True:
                i = 10
        b = '\n            def foo(((a, b), c), d, *vargs, **kwargs) -> z:\n                x = 5'
        a = '\n            def foo(xxx_todo_changeme, d, *vargs, **kwargs) -> z:\n                ((a, b), c) = xxx_todo_changeme\n                x = 5'
        self.check(b, a)

    def test_multi_1(self):
        if False:
            for i in range(10):
                print('nop')
        b = '\n            def foo(((a, b), c), (d, e, f)) -> z:\n                x = 5'
        a = '\n            def foo(xxx_todo_changeme, xxx_todo_changeme1) -> z:\n                ((a, b), c) = xxx_todo_changeme\n                (d, e, f) = xxx_todo_changeme1\n                x = 5'
        self.check(b, a)

    def test_multi_2(self):
        if False:
            print('Hello World!')
        b = '\n            def foo(x, ((a, b), c), d, (e, f, g), y) -> z:\n                x = 5'
        a = '\n            def foo(x, xxx_todo_changeme, d, xxx_todo_changeme1, y) -> z:\n                ((a, b), c) = xxx_todo_changeme\n                (e, f, g) = xxx_todo_changeme1\n                x = 5'
        self.check(b, a)

    def test_docstring(self):
        if False:
            for i in range(10):
                print('nop')
        b = '\n            def foo(((a, b), c), (d, e, f)) -> z:\n                "foo foo foo foo"\n                x = 5'
        a = '\n            def foo(xxx_todo_changeme, xxx_todo_changeme1) -> z:\n                "foo foo foo foo"\n                ((a, b), c) = xxx_todo_changeme\n                (d, e, f) = xxx_todo_changeme1\n                x = 5'
        self.check(b, a)

    def test_lambda_no_change(self):
        if False:
            print('Hello World!')
        s = 'lambda x: x + 5'
        self.unchanged(s)

    def test_lambda_parens_single_arg(self):
        if False:
            i = 10
            return i + 15
        b = 'lambda (x): x + 5'
        a = 'lambda x: x + 5'
        self.check(b, a)
        b = 'lambda(x): x + 5'
        a = 'lambda x: x + 5'
        self.check(b, a)
        b = 'lambda ((((x)))): x + 5'
        a = 'lambda x: x + 5'
        self.check(b, a)
        b = 'lambda((((x)))): x + 5'
        a = 'lambda x: x + 5'
        self.check(b, a)

    def test_lambda_simple(self):
        if False:
            print('Hello World!')
        b = 'lambda (x, y): x + f(y)'
        a = 'lambda x_y: x_y[0] + f(x_y[1])'
        self.check(b, a)
        b = 'lambda(x, y): x + f(y)'
        a = 'lambda x_y: x_y[0] + f(x_y[1])'
        self.check(b, a)
        b = 'lambda (((x, y))): x + f(y)'
        a = 'lambda x_y: x_y[0] + f(x_y[1])'
        self.check(b, a)
        b = 'lambda(((x, y))): x + f(y)'
        a = 'lambda x_y: x_y[0] + f(x_y[1])'
        self.check(b, a)

    def test_lambda_one_tuple(self):
        if False:
            return 10
        b = 'lambda (x,): x + f(x)'
        a = 'lambda x1: x1[0] + f(x1[0])'
        self.check(b, a)
        b = 'lambda (((x,))): x + f(x)'
        a = 'lambda x1: x1[0] + f(x1[0])'
        self.check(b, a)

    def test_lambda_simple_multi_use(self):
        if False:
            return 10
        b = 'lambda (x, y): x + x + f(x) + x'
        a = 'lambda x_y: x_y[0] + x_y[0] + f(x_y[0]) + x_y[0]'
        self.check(b, a)

    def test_lambda_simple_reverse(self):
        if False:
            return 10
        b = 'lambda (x, y): y + x'
        a = 'lambda x_y: x_y[1] + x_y[0]'
        self.check(b, a)

    def test_lambda_nested(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'lambda (x, (y, z)): x + y + z'
        a = 'lambda x_y_z: x_y_z[0] + x_y_z[1][0] + x_y_z[1][1]'
        self.check(b, a)
        b = 'lambda (((x, (y, z)))): x + y + z'
        a = 'lambda x_y_z: x_y_z[0] + x_y_z[1][0] + x_y_z[1][1]'
        self.check(b, a)

    def test_lambda_nested_multi_use(self):
        if False:
            i = 10
            return i + 15
        b = 'lambda (x, (y, z)): x + y + f(y)'
        a = 'lambda x_y_z: x_y_z[0] + x_y_z[1][0] + f(x_y_z[1][0])'
        self.check(b, a)

class Test_methodattrs(FixerTestCase):
    fixer = 'methodattrs'
    attrs = ['func', 'self', 'class']

    def test(self):
        if False:
            print('Hello World!')
        for attr in self.attrs:
            b = 'a.im_%s' % attr
            if attr == 'class':
                a = 'a.__self__.__class__'
            else:
                a = 'a.__%s__' % attr
            self.check(b, a)
            b = 'self.foo.im_%s.foo_bar' % attr
            if attr == 'class':
                a = 'self.foo.__self__.__class__.foo_bar'
            else:
                a = 'self.foo.__%s__.foo_bar' % attr
            self.check(b, a)

    def test_unchanged(self):
        if False:
            i = 10
            return i + 15
        for attr in self.attrs:
            s = 'foo(im_%s + 5)' % attr
            self.unchanged(s)
            s = 'f(foo.__%s__)' % attr
            self.unchanged(s)
            s = 'f(foo.__%s__.foo)' % attr
            self.unchanged(s)

class Test_next(FixerTestCase):
    fixer = 'next'

    def test_1(self):
        if False:
            i = 10
            return i + 15
        b = 'it.next()'
        a = 'next(it)'
        self.check(b, a)

    def test_2(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'a.b.c.d.next()'
        a = 'next(a.b.c.d)'
        self.check(b, a)

    def test_3(self):
        if False:
            return 10
        b = '(a + b).next()'
        a = 'next((a + b))'
        self.check(b, a)

    def test_4(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'a().next()'
        a = 'next(a())'
        self.check(b, a)

    def test_5(self):
        if False:
            i = 10
            return i + 15
        b = 'a().next() + b'
        a = 'next(a()) + b'
        self.check(b, a)

    def test_6(self):
        if False:
            return 10
        b = 'c(      a().next() + b)'
        a = 'c(      next(a()) + b)'
        self.check(b, a)

    def test_prefix_preservation_1(self):
        if False:
            while True:
                i = 10
        b = '\n            for a in b:\n                foo(a)\n                a.next()\n            '
        a = '\n            for a in b:\n                foo(a)\n                next(a)\n            '
        self.check(b, a)

    def test_prefix_preservation_2(self):
        if False:
            i = 10
            return i + 15
        b = '\n            for a in b:\n                foo(a) # abc\n                # def\n                a.next()\n            '
        a = '\n            for a in b:\n                foo(a) # abc\n                # def\n                next(a)\n            '
        self.check(b, a)

    def test_prefix_preservation_3(self):
        if False:
            return 10
        b = '\n            next = 5\n            for a in b:\n                foo(a)\n                a.next()\n            '
        a = '\n            next = 5\n            for a in b:\n                foo(a)\n                a.__next__()\n            '
        self.check(b, a, ignore_warnings=True)

    def test_prefix_preservation_4(self):
        if False:
            print('Hello World!')
        b = '\n            next = 5\n            for a in b:\n                foo(a) # abc\n                # def\n                a.next()\n            '
        a = '\n            next = 5\n            for a in b:\n                foo(a) # abc\n                # def\n                a.__next__()\n            '
        self.check(b, a, ignore_warnings=True)

    def test_prefix_preservation_5(self):
        if False:
            while True:
                i = 10
        b = '\n            next = 5\n            for a in b:\n                foo(foo(a), # abc\n                    a.next())\n            '
        a = '\n            next = 5\n            for a in b:\n                foo(foo(a), # abc\n                    a.__next__())\n            '
        self.check(b, a, ignore_warnings=True)

    def test_prefix_preservation_6(self):
        if False:
            print('Hello World!')
        b = '\n            for a in b:\n                foo(foo(a), # abc\n                    a.next())\n            '
        a = '\n            for a in b:\n                foo(foo(a), # abc\n                    next(a))\n            '
        self.check(b, a)

    def test_method_1(self):
        if False:
            i = 10
            return i + 15
        b = '\n            class A:\n                def next(self):\n                    pass\n            '
        a = '\n            class A:\n                def __next__(self):\n                    pass\n            '
        self.check(b, a)

    def test_method_2(self):
        if False:
            i = 10
            return i + 15
        b = '\n            class A(object):\n                def next(self):\n                    pass\n            '
        a = '\n            class A(object):\n                def __next__(self):\n                    pass\n            '
        self.check(b, a)

    def test_method_3(self):
        if False:
            print('Hello World!')
        b = '\n            class A:\n                def next(x):\n                    pass\n            '
        a = '\n            class A:\n                def __next__(x):\n                    pass\n            '
        self.check(b, a)

    def test_method_4(self):
        if False:
            print('Hello World!')
        b = '\n            class A:\n                def __init__(self, foo):\n                    self.foo = foo\n\n                def next(self):\n                    pass\n\n                def __iter__(self):\n                    return self\n            '
        a = '\n            class A:\n                def __init__(self, foo):\n                    self.foo = foo\n\n                def __next__(self):\n                    pass\n\n                def __iter__(self):\n                    return self\n            '
        self.check(b, a)

    def test_method_unchanged(self):
        if False:
            for i in range(10):
                print('nop')
        s = '\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.unchanged(s)

    def test_shadowing_assign_simple(self):
        if False:
            return 10
        s = '\n            next = foo\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_assign_tuple_1(self):
        if False:
            while True:
                i = 10
        s = '\n            (next, a) = foo\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_assign_tuple_2(self):
        if False:
            print('Hello World!')
        s = '\n            (a, (b, (next, c)), a) = foo\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_assign_list_1(self):
        if False:
            return 10
        s = '\n            [next, a] = foo\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_assign_list_2(self):
        if False:
            i = 10
            return i + 15
        s = '\n            [a, [b, [next, c]], a] = foo\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_builtin_assign(self):
        if False:
            print('Hello World!')
        s = '\n            def foo():\n                __builtin__.next = foo\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_builtin_assign_in_tuple(self):
        if False:
            return 10
        s = '\n            def foo():\n                (a, __builtin__.next) = foo\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_builtin_assign_in_list(self):
        if False:
            return 10
        s = '\n            def foo():\n                [a, __builtin__.next] = foo\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_assign_to_next(self):
        if False:
            print('Hello World!')
        s = '\n            def foo():\n                A.next = foo\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.unchanged(s)

    def test_assign_to_next_in_tuple(self):
        if False:
            print('Hello World!')
        s = '\n            def foo():\n                (a, A.next) = foo\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.unchanged(s)

    def test_assign_to_next_in_list(self):
        if False:
            i = 10
            return i + 15
        s = '\n            def foo():\n                [a, A.next] = foo\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.unchanged(s)

    def test_shadowing_import_1(self):
        if False:
            for i in range(10):
                print('nop')
        s = '\n            import foo.bar as next\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_import_2(self):
        if False:
            return 10
        s = '\n            import bar, bar.foo as next\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_import_3(self):
        if False:
            print('Hello World!')
        s = '\n            import bar, bar.foo as next, baz\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_import_from_1(self):
        if False:
            while True:
                i = 10
        s = '\n            from x import next\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_import_from_2(self):
        if False:
            print('Hello World!')
        s = '\n            from x.a import next\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_import_from_3(self):
        if False:
            return 10
        s = '\n            from x import a, next, b\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_import_from_4(self):
        if False:
            print('Hello World!')
        s = '\n            from x.a import a, next, b\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_funcdef_1(self):
        if False:
            while True:
                i = 10
        s = '\n            def next(a):\n                pass\n\n            class A:\n                def next(self, a, b):\n                    pass\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_funcdef_2(self):
        if False:
            return 10
        b = '\n            def next(a):\n                pass\n\n            class A:\n                def next(self):\n                    pass\n\n            it.next()\n            '
        a = '\n            def next(a):\n                pass\n\n            class A:\n                def __next__(self):\n                    pass\n\n            it.__next__()\n            '
        self.warns(b, a, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_global_1(self):
        if False:
            while True:
                i = 10
        s = '\n            def f():\n                global next\n                next = 5\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_global_2(self):
        if False:
            print('Hello World!')
        s = '\n            def f():\n                global a, next, b\n                next = 5\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_for_simple(self):
        if False:
            i = 10
            return i + 15
        s = '\n            for next in it():\n                pass\n\n            b = 5\n            c = 6\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_for_tuple_1(self):
        if False:
            return 10
        s = '\n            for next, b in it():\n                pass\n\n            b = 5\n            c = 6\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_shadowing_for_tuple_2(self):
        if False:
            while True:
                i = 10
        s = '\n            for a, (next, c), b in it():\n                pass\n\n            b = 5\n            c = 6\n            '
        self.warns_unchanged(s, 'Calls to builtin next() possibly shadowed')

    def test_noncall_access_1(self):
        if False:
            return 10
        b = 'gnext = g.next'
        a = 'gnext = g.__next__'
        self.check(b, a)

    def test_noncall_access_2(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'f(g.next + 5)'
        a = 'f(g.__next__ + 5)'
        self.check(b, a)

    def test_noncall_access_3(self):
        if False:
            while True:
                i = 10
        b = 'f(g().next + 5)'
        a = 'f(g().__next__ + 5)'
        self.check(b, a)

class Test_nonzero(FixerTestCase):
    fixer = 'nonzero'

    def test_1(self):
        if False:
            return 10
        b = '\n            class A:\n                def __nonzero__(self):\n                    pass\n            '
        a = '\n            class A:\n                def __bool__(self):\n                    pass\n            '
        self.check(b, a)

    def test_2(self):
        if False:
            i = 10
            return i + 15
        b = '\n            class A(object):\n                def __nonzero__(self):\n                    pass\n            '
        a = '\n            class A(object):\n                def __bool__(self):\n                    pass\n            '
        self.check(b, a)

    def test_unchanged_1(self):
        if False:
            while True:
                i = 10
        s = '\n            class A(object):\n                def __bool__(self):\n                    pass\n            '
        self.unchanged(s)

    def test_unchanged_2(self):
        if False:
            while True:
                i = 10
        s = '\n            class A(object):\n                def __nonzero__(self, a):\n                    pass\n            '
        self.unchanged(s)

    def test_unchanged_func(self):
        if False:
            i = 10
            return i + 15
        s = '\n            def __nonzero__(self):\n                pass\n            '
        self.unchanged(s)

class Test_numliterals(FixerTestCase):
    fixer = 'numliterals'

    def test_octal_1(self):
        if False:
            while True:
                i = 10
        b = '0755'
        a = '0o755'
        self.check(b, a)

    def test_long_int_1(self):
        if False:
            while True:
                i = 10
        b = 'a = 12L'
        a = 'a = 12'
        self.check(b, a)

    def test_long_int_2(self):
        if False:
            while True:
                i = 10
        b = 'a = 12l'
        a = 'a = 12'
        self.check(b, a)

    def test_long_hex(self):
        if False:
            while True:
                i = 10
        b = 'b = 0x12l'
        a = 'b = 0x12'
        self.check(b, a)

    def test_comments_and_spacing(self):
        if False:
            i = 10
            return i + 15
        b = 'b =   0x12L'
        a = 'b =   0x12'
        self.check(b, a)
        b = 'b = 0755 # spam'
        a = 'b = 0o755 # spam'
        self.check(b, a)

    def test_unchanged_int(self):
        if False:
            return 10
        s = '5'
        self.unchanged(s)

    def test_unchanged_float(self):
        if False:
            return 10
        s = '5.0'
        self.unchanged(s)

    def test_unchanged_octal(self):
        if False:
            i = 10
            return i + 15
        s = '0o755'
        self.unchanged(s)

    def test_unchanged_hex(self):
        if False:
            i = 10
            return i + 15
        s = '0xABC'
        self.unchanged(s)

    def test_unchanged_exp(self):
        if False:
            for i in range(10):
                print('nop')
        s = '5.0e10'
        self.unchanged(s)

    def test_unchanged_complex_int(self):
        if False:
            while True:
                i = 10
        s = '5 + 4j'
        self.unchanged(s)

    def test_unchanged_complex_float(self):
        if False:
            return 10
        s = '5.4 + 4.9j'
        self.unchanged(s)

    def test_unchanged_complex_bare(self):
        if False:
            for i in range(10):
                print('nop')
        s = '4j'
        self.unchanged(s)
        s = '4.4j'
        self.unchanged(s)

class Test_renames(FixerTestCase):
    fixer = 'renames'
    modules = {'sys': ('maxint', 'maxsize')}

    def test_import_from(self):
        if False:
            print('Hello World!')
        for (mod, (old, new)) in list(self.modules.items()):
            b = 'from %s import %s' % (mod, old)
            a = 'from %s import %s' % (mod, new)
            self.check(b, a)
            s = 'from foo import %s' % old
            self.unchanged(s)

    def test_import_from_as(self):
        if False:
            while True:
                i = 10
        for (mod, (old, new)) in list(self.modules.items()):
            b = 'from %s import %s as foo_bar' % (mod, old)
            a = 'from %s import %s as foo_bar' % (mod, new)
            self.check(b, a)

    def test_import_module_usage(self):
        if False:
            for i in range(10):
                print('nop')
        for (mod, (old, new)) in list(self.modules.items()):
            b = '\n                import %s\n                foo(%s, %s.%s)\n                ' % (mod, mod, mod, old)
            a = '\n                import %s\n                foo(%s, %s.%s)\n                ' % (mod, mod, mod, new)
            self.check(b, a)

    def XXX_test_from_import_usage(self):
        if False:
            for i in range(10):
                print('nop')
        for (mod, (old, new)) in list(self.modules.items()):
            b = '\n                from %s import %s\n                foo(%s, %s)\n                ' % (mod, old, mod, old)
            a = '\n                from %s import %s\n                foo(%s, %s)\n                ' % (mod, new, mod, new)
            self.check(b, a)

class Test_unicode(FixerTestCase):
    fixer = 'unicode'

    def test_whitespace(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'unicode( x)'
        a = 'str( x)'
        self.check(b, a)
        b = ' unicode(x )'
        a = ' str(x )'
        self.check(b, a)
        b = " u'h'"
        a = " 'h'"
        self.check(b, a)

    def test_unicode_call(self):
        if False:
            while True:
                i = 10
        b = 'unicode(x, y, z)'
        a = 'str(x, y, z)'
        self.check(b, a)

    def test_unichr(self):
        if False:
            i = 10
            return i + 15
        b = "unichr(u'h')"
        a = "chr('h')"
        self.check(b, a)

    def test_unicode_literal_1(self):
        if False:
            while True:
                i = 10
        b = 'u"x"'
        a = '"x"'
        self.check(b, a)

    def test_unicode_literal_2(self):
        if False:
            for i in range(10):
                print('nop')
        b = "ur'x'"
        a = "r'x'"
        self.check(b, a)

    def test_unicode_literal_3(self):
        if False:
            for i in range(10):
                print('nop')
        b = "UR'''x''' "
        a = "R'''x''' "
        self.check(b, a)

    def test_native_literal_escape_u(self):
        if False:
            i = 10
            return i + 15
        b = "'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        a = "'\\\\\\\\u20ac\\\\U0001d121\\\\u20ac'"
        self.check(b, a)
        b = "r'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        a = "r'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        self.check(b, a)

    def test_bytes_literal_escape_u(self):
        if False:
            i = 10
            return i + 15
        b = "b'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        a = "b'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        self.check(b, a)
        b = "br'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        a = "br'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        self.check(b, a)

    def test_unicode_literal_escape_u(self):
        if False:
            return 10
        b = "u'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        a = "'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        self.check(b, a)
        b = "ur'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        a = "r'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        self.check(b, a)

    def test_native_unicode_literal_escape_u(self):
        if False:
            print('Hello World!')
        f = 'from __future__ import unicode_literals\n'
        b = f + "'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        a = f + "'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        self.check(b, a)
        b = f + "r'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        a = f + "r'\\\\\\u20ac\\U0001d121\\\\u20ac'"
        self.check(b, a)

class Test_filter(FixerTestCase):
    fixer = 'filter'

    def test_prefix_preservation(self):
        if False:
            return 10
        b = "x =   filter(    foo,     'abc'   )"
        a = "x =   list(filter(    foo,     'abc'   ))"
        self.check(b, a)
        b = "x =   filter(  None , 'abc'  )"
        a = "x =   [_f for _f in 'abc' if _f]"
        self.check(b, a)

    def test_filter_basic(self):
        if False:
            while True:
                i = 10
        b = "x = filter(None, 'abc')"
        a = "x = [_f for _f in 'abc' if _f]"
        self.check(b, a)
        b = "x = len(filter(f, 'abc'))"
        a = "x = len(list(filter(f, 'abc')))"
        self.check(b, a)
        b = 'x = filter(lambda x: x%2 == 0, range(10))'
        a = 'x = [x for x in range(10) if x%2 == 0]'
        self.check(b, a)
        b = 'x = filter(lambda (x): x%2 == 0, range(10))'
        a = 'x = [x for x in range(10) if x%2 == 0]'
        self.check(b, a)
        b = 'filter(lambda x: True if x > 2 else False, [1, 2, 3])'
        a = '[x for x in [1, 2, 3] if (True if x > 2 else False)]'
        self.check(b, a)

    def test_filter_trailers(self):
        if False:
            print('Hello World!')
        b = "x = filter(None, 'abc')[0]"
        a = "x = [_f for _f in 'abc' if _f][0]"
        self.check(b, a)
        b = "x = len(filter(f, 'abc')[0])"
        a = "x = len(list(filter(f, 'abc'))[0])"
        self.check(b, a)
        b = 'x = filter(lambda x: x%2 == 0, range(10))[0]'
        a = 'x = [x for x in range(10) if x%2 == 0][0]'
        self.check(b, a)
        b = 'x = filter(lambda (x): x%2 == 0, range(10))[0]'
        a = 'x = [x for x in range(10) if x%2 == 0][0]'
        self.check(b, a)

    def test_filter_nochange(self):
        if False:
            print('Hello World!')
        a = "b.join(filter(f, 'abc'))"
        self.unchanged(a)
        a = "(a + foo(5)).join(filter(f, 'abc'))"
        self.unchanged(a)
        a = "iter(filter(f, 'abc'))"
        self.unchanged(a)
        a = "list(filter(f, 'abc'))"
        self.unchanged(a)
        a = "list(filter(f, 'abc'))[0]"
        self.unchanged(a)
        a = "set(filter(f, 'abc'))"
        self.unchanged(a)
        a = "set(filter(f, 'abc')).pop()"
        self.unchanged(a)
        a = "tuple(filter(f, 'abc'))"
        self.unchanged(a)
        a = "any(filter(f, 'abc'))"
        self.unchanged(a)
        a = "all(filter(f, 'abc'))"
        self.unchanged(a)
        a = "sum(filter(f, 'abc'))"
        self.unchanged(a)
        a = "sorted(filter(f, 'abc'))"
        self.unchanged(a)
        a = "sorted(filter(f, 'abc'), key=blah)"
        self.unchanged(a)
        a = "sorted(filter(f, 'abc'), key=blah)[0]"
        self.unchanged(a)
        a = "enumerate(filter(f, 'abc'))"
        self.unchanged(a)
        a = "enumerate(filter(f, 'abc'), start=1)"
        self.unchanged(a)
        a = "for i in filter(f, 'abc'): pass"
        self.unchanged(a)
        a = "[x for x in filter(f, 'abc')]"
        self.unchanged(a)
        a = "(x for x in filter(f, 'abc'))"
        self.unchanged(a)

    def test_future_builtins(self):
        if False:
            while True:
                i = 10
        a = "from future_builtins import spam, filter; filter(f, 'ham')"
        self.unchanged(a)
        b = "from future_builtins import spam; x = filter(f, 'abc')"
        a = "from future_builtins import spam; x = list(filter(f, 'abc'))"
        self.check(b, a)
        a = "from future_builtins import *; filter(f, 'ham')"
        self.unchanged(a)

class Test_map(FixerTestCase):
    fixer = 'map'

    def check(self, b, a):
        if False:
            while True:
                i = 10
        self.unchanged('from future_builtins import map; ' + b, a)
        super(Test_map, self).check(b, a)

    def test_prefix_preservation(self):
        if False:
            print('Hello World!')
        b = "x =    map(   f,    'abc'   )"
        a = "x =    list(map(   f,    'abc'   ))"
        self.check(b, a)

    def test_map_trailers(self):
        if False:
            for i in range(10):
                print('nop')
        b = "x = map(f, 'abc')[0]"
        a = "x = list(map(f, 'abc'))[0]"
        self.check(b, a)
        b = 'x = map(None, l)[0]'
        a = 'x = list(l)[0]'
        self.check(b, a)
        b = 'x = map(lambda x:x, l)[0]'
        a = 'x = [x for x in l][0]'
        self.check(b, a)
        b = "x = map(f, 'abc')[0][1]"
        a = "x = list(map(f, 'abc'))[0][1]"
        self.check(b, a)

    def test_trailing_comment(self):
        if False:
            print('Hello World!')
        b = "x = map(f, 'abc')   #   foo"
        a = "x = list(map(f, 'abc'))   #   foo"
        self.check(b, a)

    def test_None_with_multiple_arguments(self):
        if False:
            print('Hello World!')
        s = 'x = map(None, a, b, c)'
        self.warns_unchanged(s, 'cannot convert map(None, ...) with multiple arguments')

    def test_map_basic(self):
        if False:
            while True:
                i = 10
        b = "x = map(f, 'abc')"
        a = "x = list(map(f, 'abc'))"
        self.check(b, a)
        b = "x = len(map(f, 'abc', 'def'))"
        a = "x = len(list(map(f, 'abc', 'def')))"
        self.check(b, a)
        b = "x = map(None, 'abc')"
        a = "x = list('abc')"
        self.check(b, a)
        b = 'x = map(lambda x: x+1, range(4))'
        a = 'x = [x+1 for x in range(4)]'
        self.check(b, a)
        b = 'x = map(lambda (x): x+1, range(4))'
        a = 'x = [x+1 for x in range(4)]'
        self.check(b, a)
        b = '\n            foo()\n            # foo\n            map(f, x)\n            '
        a = '\n            foo()\n            # foo\n            list(map(f, x))\n            '
        self.warns(b, a, 'You should use a for loop here')

    def test_map_nochange(self):
        if False:
            for i in range(10):
                print('nop')
        a = "b.join(map(f, 'abc'))"
        self.unchanged(a)
        a = "(a + foo(5)).join(map(f, 'abc'))"
        self.unchanged(a)
        a = "iter(map(f, 'abc'))"
        self.unchanged(a)
        a = "list(map(f, 'abc'))"
        self.unchanged(a)
        a = "list(map(f, 'abc'))[0]"
        self.unchanged(a)
        a = "set(map(f, 'abc'))"
        self.unchanged(a)
        a = "set(map(f, 'abc')).pop()"
        self.unchanged(a)
        a = "tuple(map(f, 'abc'))"
        self.unchanged(a)
        a = "any(map(f, 'abc'))"
        self.unchanged(a)
        a = "all(map(f, 'abc'))"
        self.unchanged(a)
        a = "sum(map(f, 'abc'))"
        self.unchanged(a)
        a = "sorted(map(f, 'abc'))"
        self.unchanged(a)
        a = "sorted(map(f, 'abc'), key=blah)"
        self.unchanged(a)
        a = "sorted(map(f, 'abc'), key=blah)[0]"
        self.unchanged(a)
        a = "enumerate(map(f, 'abc'))"
        self.unchanged(a)
        a = "enumerate(map(f, 'abc'), start=1)"
        self.unchanged(a)
        a = "for i in map(f, 'abc'): pass"
        self.unchanged(a)
        a = "[x for x in map(f, 'abc')]"
        self.unchanged(a)
        a = "(x for x in map(f, 'abc'))"
        self.unchanged(a)

    def test_future_builtins(self):
        if False:
            return 10
        a = "from future_builtins import spam, map, eggs; map(f, 'ham')"
        self.unchanged(a)
        b = "from future_builtins import spam, eggs; x = map(f, 'abc')"
        a = "from future_builtins import spam, eggs; x = list(map(f, 'abc'))"
        self.check(b, a)
        a = "from future_builtins import *; map(f, 'ham')"
        self.unchanged(a)

class Test_zip(FixerTestCase):
    fixer = 'zip'

    def check(self, b, a):
        if False:
            print('Hello World!')
        self.unchanged('from future_builtins import zip; ' + b, a)
        super(Test_zip, self).check(b, a)

    def test_zip_basic(self):
        if False:
            while True:
                i = 10
        b = 'x = zip()'
        a = 'x = list(zip())'
        self.check(b, a)
        b = 'x = zip(a, b, c)'
        a = 'x = list(zip(a, b, c))'
        self.check(b, a)
        b = 'x = len(zip(a, b))'
        a = 'x = len(list(zip(a, b)))'
        self.check(b, a)

    def test_zip_trailers(self):
        if False:
            print('Hello World!')
        b = 'x = zip(a, b, c)[0]'
        a = 'x = list(zip(a, b, c))[0]'
        self.check(b, a)
        b = 'x = zip(a, b, c)[0][1]'
        a = 'x = list(zip(a, b, c))[0][1]'
        self.check(b, a)

    def test_zip_nochange(self):
        if False:
            i = 10
            return i + 15
        a = 'b.join(zip(a, b))'
        self.unchanged(a)
        a = '(a + foo(5)).join(zip(a, b))'
        self.unchanged(a)
        a = 'iter(zip(a, b))'
        self.unchanged(a)
        a = 'list(zip(a, b))'
        self.unchanged(a)
        a = 'list(zip(a, b))[0]'
        self.unchanged(a)
        a = 'set(zip(a, b))'
        self.unchanged(a)
        a = 'set(zip(a, b)).pop()'
        self.unchanged(a)
        a = 'tuple(zip(a, b))'
        self.unchanged(a)
        a = 'any(zip(a, b))'
        self.unchanged(a)
        a = 'all(zip(a, b))'
        self.unchanged(a)
        a = 'sum(zip(a, b))'
        self.unchanged(a)
        a = 'sorted(zip(a, b))'
        self.unchanged(a)
        a = 'sorted(zip(a, b), key=blah)'
        self.unchanged(a)
        a = 'sorted(zip(a, b), key=blah)[0]'
        self.unchanged(a)
        a = 'enumerate(zip(a, b))'
        self.unchanged(a)
        a = 'enumerate(zip(a, b), start=1)'
        self.unchanged(a)
        a = 'for i in zip(a, b): pass'
        self.unchanged(a)
        a = '[x for x in zip(a, b)]'
        self.unchanged(a)
        a = '(x for x in zip(a, b))'
        self.unchanged(a)

    def test_future_builtins(self):
        if False:
            i = 10
            return i + 15
        a = 'from future_builtins import spam, zip, eggs; zip(a, b)'
        self.unchanged(a)
        b = 'from future_builtins import spam, eggs; x = zip(a, b)'
        a = 'from future_builtins import spam, eggs; x = list(zip(a, b))'
        self.check(b, a)
        a = 'from future_builtins import *; zip(a, b)'
        self.unchanged(a)

class Test_standarderror(FixerTestCase):
    fixer = 'standarderror'

    def test(self):
        if False:
            print('Hello World!')
        b = 'x =    StandardError()'
        a = 'x =    Exception()'
        self.check(b, a)
        b = 'x = StandardError(a, b, c)'
        a = 'x = Exception(a, b, c)'
        self.check(b, a)
        b = 'f(2 + StandardError(a, b, c))'
        a = 'f(2 + Exception(a, b, c))'
        self.check(b, a)

class Test_types(FixerTestCase):
    fixer = 'types'

    def test_basic_types_convert(self):
        if False:
            print('Hello World!')
        b = 'types.StringType'
        a = 'bytes'
        self.check(b, a)
        b = 'types.DictType'
        a = 'dict'
        self.check(b, a)
        b = 'types . IntType'
        a = 'int'
        self.check(b, a)
        b = 'types.ListType'
        a = 'list'
        self.check(b, a)
        b = 'types.LongType'
        a = 'int'
        self.check(b, a)
        b = 'types.NoneType'
        a = 'type(None)'
        self.check(b, a)
        b = 'types.StringTypes'
        a = '(str,)'
        self.check(b, a)

class Test_idioms(FixerTestCase):
    fixer = 'idioms'

    def test_while(self):
        if False:
            while True:
                i = 10
        b = 'while 1: foo()'
        a = 'while True: foo()'
        self.check(b, a)
        b = 'while   1: foo()'
        a = 'while   True: foo()'
        self.check(b, a)
        b = '\n            while 1:\n                foo()\n            '
        a = '\n            while True:\n                foo()\n            '
        self.check(b, a)

    def test_while_unchanged(self):
        if False:
            i = 10
            return i + 15
        s = 'while 11: foo()'
        self.unchanged(s)
        s = 'while 0: foo()'
        self.unchanged(s)
        s = 'while foo(): foo()'
        self.unchanged(s)
        s = 'while []: foo()'
        self.unchanged(s)

    def test_eq_simple(self):
        if False:
            while True:
                i = 10
        b = 'type(x) == T'
        a = 'isinstance(x, T)'
        self.check(b, a)
        b = 'if   type(x) == T: pass'
        a = 'if   isinstance(x, T): pass'
        self.check(b, a)

    def test_eq_reverse(self):
        if False:
            print('Hello World!')
        b = 'T == type(x)'
        a = 'isinstance(x, T)'
        self.check(b, a)
        b = 'if   T == type(x): pass'
        a = 'if   isinstance(x, T): pass'
        self.check(b, a)

    def test_eq_expression(self):
        if False:
            i = 10
            return i + 15
        b = "type(x+y) == d.get('T')"
        a = "isinstance(x+y, d.get('T'))"
        self.check(b, a)
        b = "type(   x  +  y) == d.get('T')"
        a = "isinstance(x  +  y, d.get('T'))"
        self.check(b, a)

    def test_is_simple(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'type(x) is T'
        a = 'isinstance(x, T)'
        self.check(b, a)
        b = 'if   type(x) is T: pass'
        a = 'if   isinstance(x, T): pass'
        self.check(b, a)

    def test_is_reverse(self):
        if False:
            while True:
                i = 10
        b = 'T is type(x)'
        a = 'isinstance(x, T)'
        self.check(b, a)
        b = 'if   T is type(x): pass'
        a = 'if   isinstance(x, T): pass'
        self.check(b, a)

    def test_is_expression(self):
        if False:
            print('Hello World!')
        b = "type(x+y) is d.get('T')"
        a = "isinstance(x+y, d.get('T'))"
        self.check(b, a)
        b = "type(   x  +  y) is d.get('T')"
        a = "isinstance(x  +  y, d.get('T'))"
        self.check(b, a)

    def test_is_not_simple(self):
        if False:
            return 10
        b = 'type(x) is not T'
        a = 'not isinstance(x, T)'
        self.check(b, a)
        b = 'if   type(x) is not T: pass'
        a = 'if   not isinstance(x, T): pass'
        self.check(b, a)

    def test_is_not_reverse(self):
        if False:
            while True:
                i = 10
        b = 'T is not type(x)'
        a = 'not isinstance(x, T)'
        self.check(b, a)
        b = 'if   T is not type(x): pass'
        a = 'if   not isinstance(x, T): pass'
        self.check(b, a)

    def test_is_not_expression(self):
        if False:
            return 10
        b = "type(x+y) is not d.get('T')"
        a = "not isinstance(x+y, d.get('T'))"
        self.check(b, a)
        b = "type(   x  +  y) is not d.get('T')"
        a = "not isinstance(x  +  y, d.get('T'))"
        self.check(b, a)

    def test_ne_simple(self):
        if False:
            print('Hello World!')
        b = 'type(x) != T'
        a = 'not isinstance(x, T)'
        self.check(b, a)
        b = 'if   type(x) != T: pass'
        a = 'if   not isinstance(x, T): pass'
        self.check(b, a)

    def test_ne_reverse(self):
        if False:
            while True:
                i = 10
        b = 'T != type(x)'
        a = 'not isinstance(x, T)'
        self.check(b, a)
        b = 'if   T != type(x): pass'
        a = 'if   not isinstance(x, T): pass'
        self.check(b, a)

    def test_ne_expression(self):
        if False:
            i = 10
            return i + 15
        b = "type(x+y) != d.get('T')"
        a = "not isinstance(x+y, d.get('T'))"
        self.check(b, a)
        b = "type(   x  +  y) != d.get('T')"
        a = "not isinstance(x  +  y, d.get('T'))"
        self.check(b, a)

    def test_type_unchanged(self):
        if False:
            return 10
        a = 'type(x).__name__'
        self.unchanged(a)

    def test_sort_list_call(self):
        if False:
            return 10
        b = '\n            v = list(t)\n            v.sort()\n            foo(v)\n            '
        a = '\n            v = sorted(t)\n            foo(v)\n            '
        self.check(b, a)
        b = '\n            v = list(foo(b) + d)\n            v.sort()\n            foo(v)\n            '
        a = '\n            v = sorted(foo(b) + d)\n            foo(v)\n            '
        self.check(b, a)
        b = '\n            while x:\n                v = list(t)\n                v.sort()\n                foo(v)\n            '
        a = '\n            while x:\n                v = sorted(t)\n                foo(v)\n            '
        self.check(b, a)
        b = '\n            v = list(t)\n            # foo\n            v.sort()\n            foo(v)\n            '
        a = '\n            v = sorted(t)\n            # foo\n            foo(v)\n            '
        self.check(b, a)
        b = '\n            v = list(   t)\n            v.sort()\n            foo(v)\n            '
        a = '\n            v = sorted(   t)\n            foo(v)\n            '
        self.check(b, a)
        b = '\n            try:\n                m = list(s)\n                m.sort()\n            except: pass\n            '
        a = '\n            try:\n                m = sorted(s)\n            except: pass\n            '
        self.check(b, a)
        b = '\n            try:\n                m = list(s)\n                # foo\n                m.sort()\n            except: pass\n            '
        a = '\n            try:\n                m = sorted(s)\n                # foo\n            except: pass\n            '
        self.check(b, a)
        b = '\n            m = list(s)\n            # more comments\n            m.sort()'
        a = '\n            m = sorted(s)\n            # more comments'
        self.check(b, a)

    def test_sort_simple_expr(self):
        if False:
            return 10
        b = '\n            v = t\n            v.sort()\n            foo(v)\n            '
        a = '\n            v = sorted(t)\n            foo(v)\n            '
        self.check(b, a)
        b = '\n            v = foo(b)\n            v.sort()\n            foo(v)\n            '
        a = '\n            v = sorted(foo(b))\n            foo(v)\n            '
        self.check(b, a)
        b = '\n            v = b.keys()\n            v.sort()\n            foo(v)\n            '
        a = '\n            v = sorted(b.keys())\n            foo(v)\n            '
        self.check(b, a)
        b = '\n            v = foo(b) + d\n            v.sort()\n            foo(v)\n            '
        a = '\n            v = sorted(foo(b) + d)\n            foo(v)\n            '
        self.check(b, a)
        b = '\n            while x:\n                v = t\n                v.sort()\n                foo(v)\n            '
        a = '\n            while x:\n                v = sorted(t)\n                foo(v)\n            '
        self.check(b, a)
        b = '\n            v = t\n            # foo\n            v.sort()\n            foo(v)\n            '
        a = '\n            v = sorted(t)\n            # foo\n            foo(v)\n            '
        self.check(b, a)
        b = '\n            v =   t\n            v.sort()\n            foo(v)\n            '
        a = '\n            v =   sorted(t)\n            foo(v)\n            '
        self.check(b, a)

    def test_sort_unchanged(self):
        if False:
            for i in range(10):
                print('nop')
        s = '\n            v = list(t)\n            w.sort()\n            foo(w)\n            '
        self.unchanged(s)
        s = '\n            v = list(t)\n            v.sort(u)\n            foo(v)\n            '
        self.unchanged(s)

class Test_basestring(FixerTestCase):
    fixer = 'basestring'

    def test_basestring(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'isinstance(x, basestring)'
        a = 'isinstance(x, str)'
        self.check(b, a)

class Test_buffer(FixerTestCase):
    fixer = 'buffer'

    def test_buffer(self):
        if False:
            return 10
        b = 'x = buffer(y)'
        a = 'x = memoryview(y)'
        self.check(b, a)

    def test_slicing(self):
        if False:
            while True:
                i = 10
        b = 'buffer(y)[4:5]'
        a = 'memoryview(y)[4:5]'
        self.check(b, a)

class Test_future(FixerTestCase):
    fixer = 'future'

    def test_future(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'from __future__ import braces'
        a = ''
        self.check(b, a)
        b = '# comment\nfrom __future__ import braces'
        a = '# comment\n'
        self.check(b, a)
        b = 'from __future__ import braces\n# comment'
        a = '\n# comment'
        self.check(b, a)

    def test_run_order(self):
        if False:
            print('Hello World!')
        self.assert_runs_after('print')

class Test_itertools(FixerTestCase):
    fixer = 'itertools'

    def checkall(self, before, after):
        if False:
            while True:
                i = 10
        for i in ('itertools.', ''):
            for f in ('map', 'filter', 'zip'):
                b = before % (i + 'i' + f)
                a = after % f
                self.check(b, a)

    def test_0(self):
        if False:
            print('Hello World!')
        b = 'itertools.izip(a, b)'
        a = 'zip(a, b)'
        self.check(b, a)

    def test_1(self):
        if False:
            for i in range(10):
                print('nop')
        b = '%s(f, a)'
        a = '%s(f, a)'
        self.checkall(b, a)

    def test_qualified(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'itertools.ifilterfalse(a, b)'
        a = 'itertools.filterfalse(a, b)'
        self.check(b, a)
        b = 'itertools.izip_longest(a, b)'
        a = 'itertools.zip_longest(a, b)'
        self.check(b, a)

    def test_2(self):
        if False:
            while True:
                i = 10
        b = 'ifilterfalse(a, b)'
        a = 'filterfalse(a, b)'
        self.check(b, a)
        b = 'izip_longest(a, b)'
        a = 'zip_longest(a, b)'
        self.check(b, a)

    def test_space_1(self):
        if False:
            return 10
        b = '    %s(f, a)'
        a = '    %s(f, a)'
        self.checkall(b, a)

    def test_space_2(self):
        if False:
            return 10
        b = '    itertools.ifilterfalse(a, b)'
        a = '    itertools.filterfalse(a, b)'
        self.check(b, a)
        b = '    itertools.izip_longest(a, b)'
        a = '    itertools.zip_longest(a, b)'
        self.check(b, a)

    def test_run_order(self):
        if False:
            while True:
                i = 10
        self.assert_runs_after('map', 'zip', 'filter')

class Test_itertools_imports(FixerTestCase):
    fixer = 'itertools_imports'

    def test_reduced(self):
        if False:
            return 10
        b = 'from itertools import imap, izip, foo'
        a = 'from itertools import foo'
        self.check(b, a)
        b = 'from itertools import bar, imap, izip, foo'
        a = 'from itertools import bar, foo'
        self.check(b, a)
        b = 'from itertools import chain, imap, izip'
        a = 'from itertools import chain'
        self.check(b, a)

    def test_comments(self):
        if False:
            return 10
        b = '#foo\nfrom itertools import imap, izip'
        a = '#foo\n'
        self.check(b, a)

    def test_none(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'from itertools import imap, izip'
        a = ''
        self.check(b, a)
        b = 'from itertools import izip'
        a = ''
        self.check(b, a)

    def test_import_as(self):
        if False:
            i = 10
            return i + 15
        b = 'from itertools import izip, bar as bang, imap'
        a = 'from itertools import bar as bang'
        self.check(b, a)
        b = 'from itertools import izip as _zip, imap, bar'
        a = 'from itertools import bar'
        self.check(b, a)
        b = 'from itertools import imap as _map'
        a = ''
        self.check(b, a)
        b = 'from itertools import imap as _map, izip as _zip'
        a = ''
        self.check(b, a)
        s = 'from itertools import bar as bang'
        self.unchanged(s)

    def test_ifilter_and_zip_longest(self):
        if False:
            while True:
                i = 10
        for name in ('filterfalse', 'zip_longest'):
            b = 'from itertools import i%s' % (name,)
            a = 'from itertools import %s' % (name,)
            self.check(b, a)
            b = 'from itertools import imap, i%s, foo' % (name,)
            a = 'from itertools import %s, foo' % (name,)
            self.check(b, a)
            b = 'from itertools import bar, i%s, foo' % (name,)
            a = 'from itertools import bar, %s, foo' % (name,)
            self.check(b, a)

    def test_import_star(self):
        if False:
            return 10
        s = 'from itertools import *'
        self.unchanged(s)

    def test_unchanged(self):
        if False:
            while True:
                i = 10
        s = 'from itertools import foo'
        self.unchanged(s)

class Test_import(FixerTestCase):
    fixer = 'import'

    def setUp(self):
        if False:
            return 10
        super(Test_import, self).setUp()
        self.files_checked = []
        self.present_files = set()
        self.always_exists = True

        def fake_exists(name):
            if False:
                while True:
                    i = 10
            self.files_checked.append(name)
            return self.always_exists or name in self.present_files
        from lib2to3.fixes import fix_import
        fix_import.exists = fake_exists

    def tearDown(self):
        if False:
            print('Hello World!')
        from lib2to3.fixes import fix_import
        fix_import.exists = os.path.exists

    def check_both(self, b, a):
        if False:
            while True:
                i = 10
        self.always_exists = True
        super(Test_import, self).check(b, a)
        self.always_exists = False
        super(Test_import, self).unchanged(b)

    def test_files_checked(self):
        if False:
            print('Hello World!')

        def p(path):
            if False:
                for i in range(10):
                    print('nop')
            return os.path.pathsep.join(path.split('/'))
        self.always_exists = False
        self.present_files = set(['__init__.py'])
        expected_extensions = ('.py', os.path.sep, '.pyc', '.so', '.sl', '.pyd')
        names_to_test = (p('/spam/eggs.py'), 'ni.py', p('../../shrubbery.py'))
        for name in names_to_test:
            self.files_checked = []
            self.filename = name
            self.unchanged('import jam')
            if os.path.dirname(name):
                name = os.path.dirname(name) + '/jam'
            else:
                name = 'jam'
            expected_checks = set((name + ext for ext in expected_extensions))
            expected_checks.add('__init__.py')
            self.assertEqual(set(self.files_checked), expected_checks)

    def test_not_in_package(self):
        if False:
            i = 10
            return i + 15
        s = 'import bar'
        self.always_exists = False
        self.present_files = set(['bar.py'])
        self.unchanged(s)

    def test_with_absolute_import_enabled(self):
        if False:
            return 10
        s = 'from __future__ import absolute_import\nimport bar'
        self.always_exists = False
        self.present_files = set(['__init__.py', 'bar.py'])
        self.unchanged(s)

    def test_in_package(self):
        if False:
            print('Hello World!')
        b = 'import bar'
        a = 'from . import bar'
        self.always_exists = False
        self.present_files = set(['__init__.py', 'bar.py'])
        self.check(b, a)

    def test_import_from_package(self):
        if False:
            return 10
        b = 'import bar'
        a = 'from . import bar'
        self.always_exists = False
        self.present_files = set(['__init__.py', 'bar' + os.path.sep])
        self.check(b, a)

    def test_already_relative_import(self):
        if False:
            i = 10
            return i + 15
        s = 'from . import bar'
        self.unchanged(s)

    def test_comments_and_indent(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'import bar # Foo'
        a = 'from . import bar # Foo'
        self.check(b, a)

    def test_from(self):
        if False:
            print('Hello World!')
        b = 'from foo import bar, baz'
        a = 'from .foo import bar, baz'
        self.check_both(b, a)
        b = 'from foo import bar'
        a = 'from .foo import bar'
        self.check_both(b, a)
        b = 'from foo import (bar, baz)'
        a = 'from .foo import (bar, baz)'
        self.check_both(b, a)

    def test_dotted_from(self):
        if False:
            return 10
        b = 'from green.eggs import ham'
        a = 'from .green.eggs import ham'
        self.check_both(b, a)

    def test_from_as(self):
        if False:
            return 10
        b = 'from green.eggs import ham as spam'
        a = 'from .green.eggs import ham as spam'
        self.check_both(b, a)

    def test_import(self):
        if False:
            return 10
        b = 'import foo'
        a = 'from . import foo'
        self.check_both(b, a)
        b = 'import foo, bar'
        a = 'from . import foo, bar'
        self.check_both(b, a)
        b = 'import foo, bar, x'
        a = 'from . import foo, bar, x'
        self.check_both(b, a)
        b = 'import x, y, z'
        a = 'from . import x, y, z'
        self.check_both(b, a)

    def test_import_as(self):
        if False:
            while True:
                i = 10
        b = 'import foo as x'
        a = 'from . import foo as x'
        self.check_both(b, a)
        b = 'import a as b, b as c, c as d'
        a = 'from . import a as b, b as c, c as d'
        self.check_both(b, a)

    def test_local_and_absolute(self):
        if False:
            print('Hello World!')
        self.always_exists = False
        self.present_files = set(['foo.py', '__init__.py'])
        s = 'import foo, bar'
        self.warns_unchanged(s, 'absolute and local imports together')

    def test_dotted_import(self):
        if False:
            return 10
        b = 'import foo.bar'
        a = 'from . import foo.bar'
        self.check_both(b, a)

    def test_dotted_import_as(self):
        if False:
            return 10
        b = 'import foo.bar as bang'
        a = 'from . import foo.bar as bang'
        self.check_both(b, a)

    def test_prefix(self):
        if False:
            return 10
        b = '\n        # prefix\n        import foo.bar\n        '
        a = '\n        # prefix\n        from . import foo.bar\n        '
        self.check_both(b, a)

class Test_set_literal(FixerTestCase):
    fixer = 'set_literal'

    def test_basic(self):
        if False:
            while True:
                i = 10
        b = 'set([1, 2, 3])'
        a = '{1, 2, 3}'
        self.check(b, a)
        b = 'set((1, 2, 3))'
        a = '{1, 2, 3}'
        self.check(b, a)
        b = 'set((1,))'
        a = '{1}'
        self.check(b, a)
        b = 'set([1])'
        self.check(b, a)
        b = 'set((a, b))'
        a = '{a, b}'
        self.check(b, a)
        b = 'set([a, b])'
        self.check(b, a)
        b = 'set((a*234, f(args=23)))'
        a = '{a*234, f(args=23)}'
        self.check(b, a)
        b = 'set([a*23, f(23)])'
        a = '{a*23, f(23)}'
        self.check(b, a)
        b = 'set([a-234**23])'
        a = '{a-234**23}'
        self.check(b, a)

    def test_listcomps(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'set([x for x in y])'
        a = '{x for x in y}'
        self.check(b, a)
        b = 'set([x for x in y if x == m])'
        a = '{x for x in y if x == m}'
        self.check(b, a)
        b = 'set([x for x in y for a in b])'
        a = '{x for x in y for a in b}'
        self.check(b, a)
        b = 'set([f(x) - 23 for x in y])'
        a = '{f(x) - 23 for x in y}'
        self.check(b, a)

    def test_whitespace(self):
        if False:
            return 10
        b = 'set( [1, 2])'
        a = '{1, 2}'
        self.check(b, a)
        b = 'set([1 ,  2])'
        a = '{1 ,  2}'
        self.check(b, a)
        b = 'set([ 1 ])'
        a = '{ 1 }'
        self.check(b, a)
        b = 'set( [1] )'
        a = '{1}'
        self.check(b, a)
        b = 'set([  1,  2  ])'
        a = '{  1,  2  }'
        self.check(b, a)
        b = 'set([x  for x in y ])'
        a = '{x  for x in y }'
        self.check(b, a)
        b = 'set(\n                   [1, 2]\n               )\n            '
        a = '{1, 2}\n'
        self.check(b, a)

    def test_comments(self):
        if False:
            print('Hello World!')
        b = 'set((1, 2)) # Hi'
        a = '{1, 2} # Hi'
        self.check(b, a)
        b = '\n            # Foo\n            set( # Bar\n               (1, 2)\n            )\n            '
        a = '\n            # Foo\n            {1, 2}\n            '
        self.check(b, a)

    def test_unchanged(self):
        if False:
            for i in range(10):
                print('nop')
        s = 'set()'
        self.unchanged(s)
        s = 'set(a)'
        self.unchanged(s)
        s = 'set(a, b, c)'
        self.unchanged(s)
        s = 'set(x for x in y)'
        self.unchanged(s)
        s = 'set(x for x in y if z)'
        self.unchanged(s)
        s = 'set(a*823-23**2 + f(23))'
        self.unchanged(s)

class Test_sys_exc(FixerTestCase):
    fixer = 'sys_exc'

    def test_0(self):
        if False:
            i = 10
            return i + 15
        b = 'sys.exc_type'
        a = 'sys.exc_info()[0]'
        self.check(b, a)

    def test_1(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'sys.exc_value'
        a = 'sys.exc_info()[1]'
        self.check(b, a)

    def test_2(self):
        if False:
            return 10
        b = 'sys.exc_traceback'
        a = 'sys.exc_info()[2]'
        self.check(b, a)

    def test_3(self):
        if False:
            i = 10
            return i + 15
        b = 'sys.exc_type # Foo'
        a = 'sys.exc_info()[0] # Foo'
        self.check(b, a)

    def test_4(self):
        if False:
            return 10
        b = 'sys.  exc_type'
        a = 'sys.  exc_info()[0]'
        self.check(b, a)

    def test_5(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'sys  .exc_type'
        a = 'sys  .exc_info()[0]'
        self.check(b, a)

class Test_paren(FixerTestCase):
    fixer = 'paren'

    def test_0(self):
        if False:
            return 10
        b = '[i for i in 1, 2 ]'
        a = '[i for i in (1, 2) ]'
        self.check(b, a)

    def test_1(self):
        if False:
            print('Hello World!')
        b = '[i for i in 1, 2, ]'
        a = '[i for i in (1, 2,) ]'
        self.check(b, a)

    def test_2(self):
        if False:
            i = 10
            return i + 15
        b = '[i for i  in     1, 2 ]'
        a = '[i for i  in     (1, 2) ]'
        self.check(b, a)

    def test_3(self):
        if False:
            i = 10
            return i + 15
        b = '[i for i in 1, 2 if i]'
        a = '[i for i in (1, 2) if i]'
        self.check(b, a)

    def test_4(self):
        if False:
            print('Hello World!')
        b = '[i for i in 1,    2    ]'
        a = '[i for i in (1,    2)    ]'
        self.check(b, a)

    def test_5(self):
        if False:
            while True:
                i = 10
        b = '(i for i in 1, 2)'
        a = '(i for i in (1, 2))'
        self.check(b, a)

    def test_6(self):
        if False:
            while True:
                i = 10
        b = '(i for i in 1   ,2   if i)'
        a = '(i for i in (1   ,2)   if i)'
        self.check(b, a)

    def test_unchanged_0(self):
        if False:
            print('Hello World!')
        s = '[i for i in (1, 2)]'
        self.unchanged(s)

    def test_unchanged_1(self):
        if False:
            while True:
                i = 10
        s = '[i for i in foo()]'
        self.unchanged(s)

    def test_unchanged_2(self):
        if False:
            for i in range(10):
                print('nop')
        s = '[i for i in (1, 2) if nothing]'
        self.unchanged(s)

    def test_unchanged_3(self):
        if False:
            for i in range(10):
                print('nop')
        s = '(i for i in (1, 2))'
        self.unchanged(s)

    def test_unchanged_4(self):
        if False:
            return 10
        s = '[i for i in m]'
        self.unchanged(s)

class Test_metaclass(FixerTestCase):
    fixer = 'metaclass'

    def test_unchanged(self):
        if False:
            while True:
                i = 10
        self.unchanged('class X(): pass')
        self.unchanged('class X(object): pass')
        self.unchanged('class X(object1, object2): pass')
        self.unchanged('class X(object1, object2, object3): pass')
        self.unchanged('class X(metaclass=Meta): pass')
        self.unchanged('class X(b, arg=23, metclass=Meta): pass')
        self.unchanged('class X(b, arg=23, metaclass=Meta, other=42): pass')
        s = '\n        class X:\n            def __metaclass__(self): pass\n        '
        self.unchanged(s)
        s = '\n        class X:\n            a[23] = 74\n        '
        self.unchanged(s)

    def test_comments(self):
        if False:
            print('Hello World!')
        b = '\n        class X:\n            # hi\n            __metaclass__ = AppleMeta\n        '
        a = '\n        class X(metaclass=AppleMeta):\n            # hi\n            pass\n        '
        self.check(b, a)
        b = '\n        class X:\n            __metaclass__ = Meta\n            # Bedtime!\n        '
        a = '\n        class X(metaclass=Meta):\n            pass\n            # Bedtime!\n        '
        self.check(b, a)

    def test_meta(self):
        if False:
            i = 10
            return i + 15
        b = '\n        class X():\n            __metaclass__ = Q\n            pass\n        '
        a = '\n        class X(metaclass=Q):\n            pass\n        '
        self.check(b, a)
        b = 'class X(object): __metaclass__ = Q'
        a = 'class X(object, metaclass=Q): pass'
        self.check(b, a)
        b = '\n        class X(object):\n            __metaclass__ = Meta\n            bar = 7\n        '
        a = '\n        class X(object, metaclass=Meta):\n            bar = 7\n        '
        self.check(b, a)
        b = '\n        class X:\n            __metaclass__ = Meta; x = 4; g = 23\n        '
        a = '\n        class X(metaclass=Meta):\n            x = 4; g = 23\n        '
        self.check(b, a)
        b = '\n        class X(object):\n            bar = 7\n            __metaclass__ = Meta\n        '
        a = '\n        class X(object, metaclass=Meta):\n            bar = 7\n        '
        self.check(b, a)
        b = '\n        class X():\n            __metaclass__ = A\n            __metaclass__ = B\n            bar = 7\n        '
        a = '\n        class X(metaclass=B):\n            bar = 7\n        '
        self.check(b, a)
        b = '\n        class X(clsA, clsB):\n            __metaclass__ = Meta\n            bar = 7\n        '
        a = '\n        class X(clsA, clsB, metaclass=Meta):\n            bar = 7\n        '
        self.check(b, a)
        b = 'class m(a, arg=23): __metaclass__ = Meta'
        a = 'class m(a, arg=23, metaclass=Meta): pass'
        self.check(b, a)
        b = '\n        class X(expression(2 + 4)):\n            __metaclass__ = Meta\n        '
        a = '\n        class X(expression(2 + 4), metaclass=Meta):\n            pass\n        '
        self.check(b, a)
        b = '\n        class X(expression(2 + 4), x**4):\n            __metaclass__ = Meta\n        '
        a = '\n        class X(expression(2 + 4), x**4, metaclass=Meta):\n            pass\n        '
        self.check(b, a)
        b = '\n        class X:\n            __metaclass__ = Meta\n            save.py = 23\n        '
        a = '\n        class X(metaclass=Meta):\n            save.py = 23\n        '
        self.check(b, a)

class Test_getcwdu(FixerTestCase):
    fixer = 'getcwdu'

    def test_basic(self):
        if False:
            while True:
                i = 10
        b = 'os.getcwdu'
        a = 'os.getcwd'
        self.check(b, a)
        b = 'os.getcwdu()'
        a = 'os.getcwd()'
        self.check(b, a)
        b = 'meth = os.getcwdu'
        a = 'meth = os.getcwd'
        self.check(b, a)
        b = 'os.getcwdu(args)'
        a = 'os.getcwd(args)'
        self.check(b, a)

    def test_comment(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'os.getcwdu() # Foo'
        a = 'os.getcwd() # Foo'
        self.check(b, a)

    def test_unchanged(self):
        if False:
            i = 10
            return i + 15
        s = 'os.getcwd()'
        self.unchanged(s)
        s = 'getcwdu()'
        self.unchanged(s)
        s = 'os.getcwdb()'
        self.unchanged(s)

    def test_indentation(self):
        if False:
            while True:
                i = 10
        b = '\n            if 1:\n                os.getcwdu()\n            '
        a = '\n            if 1:\n                os.getcwd()\n            '
        self.check(b, a)

    def test_multilation(self):
        if False:
            return 10
        b = 'os .getcwdu()'
        a = 'os .getcwd()'
        self.check(b, a)
        b = 'os.  getcwdu'
        a = 'os.  getcwd'
        self.check(b, a)
        b = 'os.getcwdu (  )'
        a = 'os.getcwd (  )'
        self.check(b, a)

class Test_operator(FixerTestCase):
    fixer = 'operator'

    def test_operator_isCallable(self):
        if False:
            while True:
                i = 10
        b = 'operator.isCallable(x)'
        a = 'callable(x)'
        self.check(b, a)

    def test_operator_sequenceIncludes(self):
        if False:
            while True:
                i = 10
        b = 'operator.sequenceIncludes(x, y)'
        a = 'operator.contains(x, y)'
        self.check(b, a)
        b = 'operator .sequenceIncludes(x, y)'
        a = 'operator .contains(x, y)'
        self.check(b, a)
        b = 'operator.  sequenceIncludes(x, y)'
        a = 'operator.  contains(x, y)'
        self.check(b, a)

    def test_operator_isSequenceType(self):
        if False:
            while True:
                i = 10
        b = 'operator.isSequenceType(x)'
        a = 'import collections.abc\nisinstance(x, collections.abc.Sequence)'
        self.check(b, a)

    def test_operator_isMappingType(self):
        if False:
            for i in range(10):
                print('nop')
        b = 'operator.isMappingType(x)'
        a = 'import collections.abc\nisinstance(x, collections.abc.Mapping)'
        self.check(b, a)

    def test_operator_isNumberType(self):
        if False:
            i = 10
            return i + 15
        b = 'operator.isNumberType(x)'
        a = 'import numbers\nisinstance(x, numbers.Number)'
        self.check(b, a)

    def test_operator_repeat(self):
        if False:
            return 10
        b = 'operator.repeat(x, n)'
        a = 'operator.mul(x, n)'
        self.check(b, a)
        b = 'operator .repeat(x, n)'
        a = 'operator .mul(x, n)'
        self.check(b, a)
        b = 'operator.  repeat(x, n)'
        a = 'operator.  mul(x, n)'
        self.check(b, a)

    def test_operator_irepeat(self):
        if False:
            while True:
                i = 10
        b = 'operator.irepeat(x, n)'
        a = 'operator.imul(x, n)'
        self.check(b, a)
        b = 'operator .irepeat(x, n)'
        a = 'operator .imul(x, n)'
        self.check(b, a)
        b = 'operator.  irepeat(x, n)'
        a = 'operator.  imul(x, n)'
        self.check(b, a)

    def test_bare_isCallable(self):
        if False:
            while True:
                i = 10
        s = 'isCallable(x)'
        t = "You should use 'callable(x)' here."
        self.warns_unchanged(s, t)

    def test_bare_sequenceIncludes(self):
        if False:
            return 10
        s = 'sequenceIncludes(x, y)'
        t = "You should use 'operator.contains(x, y)' here."
        self.warns_unchanged(s, t)

    def test_bare_operator_isSequenceType(self):
        if False:
            print('Hello World!')
        s = 'isSequenceType(z)'
        t = "You should use 'isinstance(z, collections.abc.Sequence)' here."
        self.warns_unchanged(s, t)

    def test_bare_operator_isMappingType(self):
        if False:
            print('Hello World!')
        s = 'isMappingType(x)'
        t = "You should use 'isinstance(x, collections.abc.Mapping)' here."
        self.warns_unchanged(s, t)

    def test_bare_operator_isNumberType(self):
        if False:
            for i in range(10):
                print('nop')
        s = 'isNumberType(y)'
        t = "You should use 'isinstance(y, numbers.Number)' here."
        self.warns_unchanged(s, t)

    def test_bare_operator_repeat(self):
        if False:
            print('Hello World!')
        s = 'repeat(x, n)'
        t = "You should use 'operator.mul(x, n)' here."
        self.warns_unchanged(s, t)

    def test_bare_operator_irepeat(self):
        if False:
            i = 10
            return i + 15
        s = 'irepeat(y, 187)'
        t = "You should use 'operator.imul(y, 187)' here."
        self.warns_unchanged(s, t)

class Test_exitfunc(FixerTestCase):
    fixer = 'exitfunc'

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        b = '\n            import sys\n            sys.exitfunc = my_atexit\n            '
        a = '\n            import sys\n            import atexit\n            atexit.register(my_atexit)\n            '
        self.check(b, a)

    def test_names_import(self):
        if False:
            print('Hello World!')
        b = '\n            import sys, crumbs\n            sys.exitfunc = my_func\n            '
        a = '\n            import sys, crumbs, atexit\n            atexit.register(my_func)\n            '
        self.check(b, a)

    def test_complex_expression(self):
        if False:
            while True:
                i = 10
        b = '\n            import sys\n            sys.exitfunc = do(d)/a()+complex(f=23, g=23)*expression\n            '
        a = '\n            import sys\n            import atexit\n            atexit.register(do(d)/a()+complex(f=23, g=23)*expression)\n            '
        self.check(b, a)

    def test_comments(self):
        if False:
            for i in range(10):
                print('nop')
        b = '\n            import sys # Foo\n            sys.exitfunc = f # Blah\n            '
        a = '\n            import sys\n            import atexit # Foo\n            atexit.register(f) # Blah\n            '
        self.check(b, a)
        b = '\n            import apples, sys, crumbs, larry # Pleasant comments\n            sys.exitfunc = func\n            '
        a = '\n            import apples, sys, crumbs, larry, atexit # Pleasant comments\n            atexit.register(func)\n            '
        self.check(b, a)

    def test_in_a_function(self):
        if False:
            return 10
        b = '\n            import sys\n            def f():\n                sys.exitfunc = func\n            '
        a = '\n            import sys\n            import atexit\n            def f():\n                atexit.register(func)\n             '
        self.check(b, a)

    def test_no_sys_import(self):
        if False:
            while True:
                i = 10
        b = 'sys.exitfunc = f'
        a = 'atexit.register(f)'
        msg = "Can't find sys import; Please add an atexit import at the top of your file."
        self.warns(b, a, msg)

    def test_unchanged(self):
        if False:
            while True:
                i = 10
        s = 'f(sys.exitfunc)'
        self.unchanged(s)

class Test_asserts(FixerTestCase):
    fixer = 'asserts'

    def test_deprecated_names(self):
        if False:
            return 10
        tests = [('self.assert_(True)', 'self.assertTrue(True)'), ('self.assertEquals(2, 2)', 'self.assertEqual(2, 2)'), ('self.assertNotEquals(2, 3)', 'self.assertNotEqual(2, 3)'), ('self.assertAlmostEquals(2, 3)', 'self.assertAlmostEqual(2, 3)'), ('self.assertNotAlmostEquals(2, 8)', 'self.assertNotAlmostEqual(2, 8)'), ('self.failUnlessEqual(2, 2)', 'self.assertEqual(2, 2)'), ('self.failIfEqual(2, 3)', 'self.assertNotEqual(2, 3)'), ('self.failUnlessAlmostEqual(2, 3)', 'self.assertAlmostEqual(2, 3)'), ('self.failIfAlmostEqual(2, 8)', 'self.assertNotAlmostEqual(2, 8)'), ('self.failUnless(True)', 'self.assertTrue(True)'), ('self.failUnlessRaises(foo)', 'self.assertRaises(foo)'), ('self.failIf(False)', 'self.assertFalse(False)')]
        for (b, a) in tests:
            self.check(b, a)

    def test_variants(self):
        if False:
            i = 10
            return i + 15
        b = 'eq = self.assertEquals'
        a = 'eq = self.assertEqual'
        self.check(b, a)
        b = 'self.assertEquals(2, 3, msg="fail")'
        a = 'self.assertEqual(2, 3, msg="fail")'
        self.check(b, a)
        b = 'self.assertEquals(2, 3, msg="fail") # foo'
        a = 'self.assertEqual(2, 3, msg="fail") # foo'
        self.check(b, a)
        b = 'self.assertEquals (2, 3)'
        a = 'self.assertEqual (2, 3)'
        self.check(b, a)
        b = '  self.assertEquals (2, 3)'
        a = '  self.assertEqual (2, 3)'
        self.check(b, a)
        b = 'with self.failUnlessRaises(Explosion): explode()'
        a = 'with self.assertRaises(Explosion): explode()'
        self.check(b, a)
        b = 'with self.failUnlessRaises(Explosion) as cm: explode()'
        a = 'with self.assertRaises(Explosion) as cm: explode()'
        self.check(b, a)

    def test_unchanged(self):
        if False:
            return 10
        self.unchanged('self.assertEqualsOnSaturday')
        self.unchanged('self.assertEqualsOnSaturday(3, 5)')