from test.support import verbose, run_doctest
from test.support.os_helper import EnvironmentVarGuard
import unittest
import getopt
sentinel = object()

class GetoptTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.env = EnvironmentVarGuard()
        if 'POSIXLY_CORRECT' in self.env:
            del self.env['POSIXLY_CORRECT']

    def tearDown(self):
        if False:
            print('Hello World!')
        self.env.__exit__()
        del self.env

    def assertError(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.assertRaises(getopt.GetoptError, *args, **kwargs)

    def test_short_has_arg(self):
        if False:
            while True:
                i = 10
        self.assertTrue(getopt.short_has_arg('a', 'a:'))
        self.assertFalse(getopt.short_has_arg('a', 'a'))
        self.assertError(getopt.short_has_arg, 'a', 'b')

    def test_long_has_args(self):
        if False:
            for i in range(10):
                print('nop')
        (has_arg, option) = getopt.long_has_args('abc', ['abc='])
        self.assertTrue(has_arg)
        self.assertEqual(option, 'abc')
        (has_arg, option) = getopt.long_has_args('abc', ['abc'])
        self.assertFalse(has_arg)
        self.assertEqual(option, 'abc')
        (has_arg, option) = getopt.long_has_args('abc', ['abcd'])
        self.assertFalse(has_arg)
        self.assertEqual(option, 'abcd')
        self.assertError(getopt.long_has_args, 'abc', ['def'])
        self.assertError(getopt.long_has_args, 'abc', [])
        self.assertError(getopt.long_has_args, 'abc', ['abcd', 'abcde'])

    def test_do_shorts(self):
        if False:
            print('Hello World!')
        (opts, args) = getopt.do_shorts([], 'a', 'a', [])
        self.assertEqual(opts, [('-a', '')])
        self.assertEqual(args, [])
        (opts, args) = getopt.do_shorts([], 'a1', 'a:', [])
        self.assertEqual(opts, [('-a', '1')])
        self.assertEqual(args, [])
        (opts, args) = getopt.do_shorts([], 'a', 'a:', ['1'])
        self.assertEqual(opts, [('-a', '1')])
        self.assertEqual(args, [])
        (opts, args) = getopt.do_shorts([], 'a', 'a:', ['1', '2'])
        self.assertEqual(opts, [('-a', '1')])
        self.assertEqual(args, ['2'])
        self.assertError(getopt.do_shorts, [], 'a1', 'a', [])
        self.assertError(getopt.do_shorts, [], 'a', 'a:', [])

    def test_do_longs(self):
        if False:
            return 10
        (opts, args) = getopt.do_longs([], 'abc', ['abc'], [])
        self.assertEqual(opts, [('--abc', '')])
        self.assertEqual(args, [])
        (opts, args) = getopt.do_longs([], 'abc=1', ['abc='], [])
        self.assertEqual(opts, [('--abc', '1')])
        self.assertEqual(args, [])
        (opts, args) = getopt.do_longs([], 'abc=1', ['abcd='], [])
        self.assertEqual(opts, [('--abcd', '1')])
        self.assertEqual(args, [])
        (opts, args) = getopt.do_longs([], 'abc', ['ab', 'abc', 'abcd'], [])
        self.assertEqual(opts, [('--abc', '')])
        self.assertEqual(args, [])
        (opts, args) = getopt.do_longs([], 'foo=42', ['foo-bar', 'foo='], [])
        self.assertEqual(opts, [('--foo', '42')])
        self.assertEqual(args, [])
        self.assertError(getopt.do_longs, [], 'abc=1', ['abc'], [])
        self.assertError(getopt.do_longs, [], 'abc', ['abc='], [])

    def test_getopt(self):
        if False:
            for i in range(10):
                print('nop')
        cmdline = ['-a', '1', '-b', '--alpha=2', '--beta', '-a', '3', '-a', '', '--beta', 'arg1', 'arg2']
        (opts, args) = getopt.getopt(cmdline, 'a:b', ['alpha=', 'beta'])
        self.assertEqual(opts, [('-a', '1'), ('-b', ''), ('--alpha', '2'), ('--beta', ''), ('-a', '3'), ('-a', ''), ('--beta', '')])
        self.assertEqual(args, ['arg1', 'arg2'])
        self.assertError(getopt.getopt, cmdline, 'a:b', ['alpha', 'beta'])

    def test_gnu_getopt(self):
        if False:
            for i in range(10):
                print('nop')
        cmdline = ['-a', 'arg1', '-b', '1', '--alpha', '--beta=2']
        (opts, args) = getopt.gnu_getopt(cmdline, 'ab:', ['alpha', 'beta='])
        self.assertEqual(args, ['arg1'])
        self.assertEqual(opts, [('-a', ''), ('-b', '1'), ('--alpha', ''), ('--beta', '2')])
        (opts, args) = getopt.gnu_getopt(['-a', '-', '-b', '-'], 'ab:', [])
        self.assertEqual(args, ['-'])
        self.assertEqual(opts, [('-a', ''), ('-b', '-')])
        (opts, args) = getopt.gnu_getopt(cmdline, '+ab:', ['alpha', 'beta='])
        self.assertEqual(opts, [('-a', '')])
        self.assertEqual(args, ['arg1', '-b', '1', '--alpha', '--beta=2'])
        self.env['POSIXLY_CORRECT'] = '1'
        (opts, args) = getopt.gnu_getopt(cmdline, 'ab:', ['alpha', 'beta='])
        self.assertEqual(opts, [('-a', '')])
        self.assertEqual(args, ['arg1', '-b', '1', '--alpha', '--beta=2'])

    def test_libref_examples(self):
        if False:
            for i in range(10):
                print('nop')
        s = "\n        Examples from the Library Reference:  Doc/lib/libgetopt.tex\n\n        An example using only Unix style options:\n\n\n        >>> import getopt\n        >>> args = '-a -b -cfoo -d bar a1 a2'.split()\n        >>> args\n        ['-a', '-b', '-cfoo', '-d', 'bar', 'a1', 'a2']\n        >>> optlist, args = getopt.getopt(args, 'abc:d:')\n        >>> optlist\n        [('-a', ''), ('-b', ''), ('-c', 'foo'), ('-d', 'bar')]\n        >>> args\n        ['a1', 'a2']\n\n        Using long option names is equally easy:\n\n\n        >>> s = '--condition=foo --testing --output-file abc.def -x a1 a2'\n        >>> args = s.split()\n        >>> args\n        ['--condition=foo', '--testing', '--output-file', 'abc.def', '-x', 'a1', 'a2']\n        >>> optlist, args = getopt.getopt(args, 'x', [\n        ...     'condition=', 'output-file=', 'testing'])\n        >>> optlist\n        [('--condition', 'foo'), ('--testing', ''), ('--output-file', 'abc.def'), ('-x', '')]\n        >>> args\n        ['a1', 'a2']\n        "
        import types
        m = types.ModuleType('libreftest', s)
        run_doctest(m, verbose)

    def test_issue4629(self):
        if False:
            return 10
        (longopts, shortopts) = getopt.getopt(['--help='], '', ['help='])
        self.assertEqual(longopts, [('--help', '')])
        (longopts, shortopts) = getopt.getopt(['--help=x'], '', ['help='])
        self.assertEqual(longopts, [('--help', 'x')])
        self.assertRaises(getopt.GetoptError, getopt.getopt, ['--help='], '', ['help'])
if __name__ == '__main__':
    unittest.main()