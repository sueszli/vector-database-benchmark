import sys
import os
import re
import copy
import unittest
from io import StringIO
from test import support
from test.support import os_helper
import optparse
from optparse import make_option, Option, TitledHelpFormatter, OptionParser, OptionGroup, SUPPRESS_USAGE, OptionError, OptionConflictError, BadOptionError, OptionValueError, Values
from optparse import _match_abbrev
from optparse import _parse_num

class InterceptedError(Exception):

    def __init__(self, error_message=None, exit_status=None, exit_message=None):
        if False:
            print('Hello World!')
        self.error_message = error_message
        self.exit_status = exit_status
        self.exit_message = exit_message

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.error_message or self.exit_message or 'intercepted error'

class InterceptingOptionParser(OptionParser):

    def exit(self, status=0, msg=None):
        if False:
            print('Hello World!')
        raise InterceptedError(exit_status=status, exit_message=msg)

    def error(self, msg):
        if False:
            return 10
        raise InterceptedError(error_message=msg)

class BaseTest(unittest.TestCase):

    def assertParseOK(self, args, expected_opts, expected_positional_args):
        if False:
            print('Hello World!')
        'Assert the options are what we expected when parsing arguments.\n\n        Otherwise, fail with a nicely formatted message.\n\n        Keyword arguments:\n        args -- A list of arguments to parse with OptionParser.\n        expected_opts -- The options expected.\n        expected_positional_args -- The positional arguments expected.\n\n        Returns the options and positional args for further testing.\n        '
        (options, positional_args) = self.parser.parse_args(args)
        optdict = vars(options)
        self.assertEqual(optdict, expected_opts, '\nOptions are %(optdict)s.\nShould be %(expected_opts)s.\nArgs were %(args)s.' % locals())
        self.assertEqual(positional_args, expected_positional_args, '\nPositional arguments are %(positional_args)s.\nShould be %(expected_positional_args)s.\nArgs were %(args)s.' % locals())
        return (options, positional_args)

    def assertRaises(self, func, args, kwargs, expected_exception, expected_message):
        if False:
            while True:
                i = 10
        '\n        Assert that the expected exception is raised when calling a\n        function, and that the right error message is included with\n        that exception.\n\n        Arguments:\n          func -- the function to call\n          args -- positional arguments to `func`\n          kwargs -- keyword arguments to `func`\n          expected_exception -- exception that should be raised\n          expected_message -- expected exception message (or pattern\n            if a compiled regex object)\n\n        Returns the exception raised for further testing.\n        '
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        try:
            func(*args, **kwargs)
        except expected_exception as err:
            actual_message = str(err)
            if isinstance(expected_message, re.Pattern):
                self.assertTrue(expected_message.search(actual_message), "expected exception message pattern:\n/%s/\nactual exception message:\n'''%s'''\n" % (expected_message.pattern, actual_message))
            else:
                self.assertEqual(actual_message, expected_message, "expected exception message:\n'''%s'''\nactual exception message:\n'''%s'''\n" % (expected_message, actual_message))
            return err
        else:
            self.fail('expected exception %(expected_exception)s not raised\ncalled %(func)r\nwith args %(args)r\nand kwargs %(kwargs)r\n' % locals())

    def assertParseFail(self, cmdline_args, expected_output):
        if False:
            i = 10
            return i + 15
        '\n        Assert the parser fails with the expected message.  Caller\n        must ensure that self.parser is an InterceptingOptionParser.\n        '
        try:
            self.parser.parse_args(cmdline_args)
        except InterceptedError as err:
            self.assertEqual(err.error_message, expected_output)
        else:
            self.assertFalse('expected parse failure')

    def assertOutput(self, cmdline_args, expected_output, expected_status=0, expected_error=None):
        if False:
            i = 10
            return i + 15
        'Assert the parser prints the expected output on stdout.'
        save_stdout = sys.stdout
        try:
            try:
                sys.stdout = StringIO()
                self.parser.parse_args(cmdline_args)
            finally:
                output = sys.stdout.getvalue()
                sys.stdout = save_stdout
        except InterceptedError as err:
            self.assertTrue(isinstance(output, str), 'expected output to be an ordinary string, not %r' % type(output))
            if output != expected_output:
                self.fail("expected: \n'''\n" + expected_output + "'''\nbut got \n'''\n" + output + "'''")
            self.assertEqual(err.exit_status, expected_status)
            self.assertEqual(err.exit_message, expected_error)
        else:
            self.assertFalse('expected parser.exit()')

    def assertTypeError(self, func, expected_message, *args):
        if False:
            print('Hello World!')
        'Assert that TypeError is raised when executing func.'
        self.assertRaises(func, args, None, TypeError, expected_message)

    def assertHelp(self, parser, expected_help):
        if False:
            return 10
        actual_help = parser.format_help()
        if actual_help != expected_help:
            raise self.failureException('help text failure; expected:\n"' + expected_help + '"; got:\n"' + actual_help + '"\n')

class TestOptionChecks(BaseTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.parser = OptionParser(usage=SUPPRESS_USAGE)

    def assertOptionError(self, expected_message, args=[], kwargs={}):
        if False:
            print('Hello World!')
        self.assertRaises(make_option, args, kwargs, OptionError, expected_message)

    def test_opt_string_empty(self):
        if False:
            return 10
        self.assertTypeError(make_option, 'at least one option string must be supplied')

    def test_opt_string_too_short(self):
        if False:
            while True:
                i = 10
        self.assertOptionError("invalid option string 'b': must be at least two characters long", ['b'])

    def test_opt_string_short_invalid(self):
        if False:
            while True:
                i = 10
        self.assertOptionError("invalid short option string '--': must be of the form -x, (x any non-dash char)", ['--'])

    def test_opt_string_long_invalid(self):
        if False:
            while True:
                i = 10
        self.assertOptionError("invalid long option string '---': must start with --, followed by non-dash", ['---'])

    def test_attr_invalid(self):
        if False:
            while True:
                i = 10
        self.assertOptionError('option -b: invalid keyword arguments: bar, foo', ['-b'], {'foo': None, 'bar': None})

    def test_action_invalid(self):
        if False:
            return 10
        self.assertOptionError("option -b: invalid action: 'foo'", ['-b'], {'action': 'foo'})

    def test_type_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertOptionError("option -b: invalid option type: 'foo'", ['-b'], {'type': 'foo'})
        self.assertOptionError("option -b: invalid option type: 'tuple'", ['-b'], {'type': tuple})

    def test_no_type_for_action(self):
        if False:
            print('Hello World!')
        self.assertOptionError("option -b: must not supply a type for action 'count'", ['-b'], {'action': 'count', 'type': 'int'})

    def test_no_choices_list(self):
        if False:
            i = 10
            return i + 15
        self.assertOptionError("option -b/--bad: must supply a list of choices for type 'choice'", ['-b', '--bad'], {'type': 'choice'})

    def test_bad_choices_list(self):
        if False:
            while True:
                i = 10
        typename = type('').__name__
        self.assertOptionError("option -b/--bad: choices must be a list of strings ('%s' supplied)" % typename, ['-b', '--bad'], {'type': 'choice', 'choices': 'bad choices'})

    def test_no_choices_for_type(self):
        if False:
            return 10
        self.assertOptionError("option -b: must not supply choices for type 'int'", ['-b'], {'type': 'int', 'choices': 'bad'})

    def test_no_const_for_action(self):
        if False:
            i = 10
            return i + 15
        self.assertOptionError("option -b: 'const' must not be supplied for action 'store'", ['-b'], {'action': 'store', 'const': 1})

    def test_no_nargs_for_action(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertOptionError("option -b: 'nargs' must not be supplied for action 'count'", ['-b'], {'action': 'count', 'nargs': 2})

    def test_callback_not_callable(self):
        if False:
            print('Hello World!')
        self.assertOptionError("option -b: callback not callable: 'foo'", ['-b'], {'action': 'callback', 'callback': 'foo'})

    def dummy(self):
        if False:
            while True:
                i = 10
        pass

    def test_callback_args_no_tuple(self):
        if False:
            while True:
                i = 10
        self.assertOptionError("option -b: callback_args, if supplied, must be a tuple: not 'foo'", ['-b'], {'action': 'callback', 'callback': self.dummy, 'callback_args': 'foo'})

    def test_callback_kwargs_no_dict(self):
        if False:
            print('Hello World!')
        self.assertOptionError("option -b: callback_kwargs, if supplied, must be a dict: not 'foo'", ['-b'], {'action': 'callback', 'callback': self.dummy, 'callback_kwargs': 'foo'})

    def test_no_callback_for_action(self):
        if False:
            i = 10
            return i + 15
        self.assertOptionError("option -b: callback supplied ('foo') for non-callback option", ['-b'], {'action': 'store', 'callback': 'foo'})

    def test_no_callback_args_for_action(self):
        if False:
            i = 10
            return i + 15
        self.assertOptionError('option -b: callback_args supplied for non-callback option', ['-b'], {'action': 'store', 'callback_args': 'foo'})

    def test_no_callback_kwargs_for_action(self):
        if False:
            print('Hello World!')
        self.assertOptionError('option -b: callback_kwargs supplied for non-callback option', ['-b'], {'action': 'store', 'callback_kwargs': 'foo'})

    def test_no_single_dash(self):
        if False:
            while True:
                i = 10
        self.assertOptionError("invalid long option string '-debug': must start with --, followed by non-dash", ['-debug'])
        self.assertOptionError("option -d: invalid long option string '-debug': must start with --, followed by non-dash", ['-d', '-debug'])
        self.assertOptionError("invalid long option string '-debug': must start with --, followed by non-dash", ['-debug', '--debug'])

class TestOptionParser(BaseTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.parser = OptionParser()
        self.parser.add_option('-v', '--verbose', '-n', '--noisy', action='store_true', dest='verbose')
        self.parser.add_option('-q', '--quiet', '--silent', action='store_false', dest='verbose')

    def test_add_option_no_Option(self):
        if False:
            return 10
        self.assertTypeError(self.parser.add_option, 'not an Option instance: None', None)

    def test_add_option_invalid_arguments(self):
        if False:
            return 10
        self.assertTypeError(self.parser.add_option, 'invalid arguments', None, None)

    def test_get_option(self):
        if False:
            i = 10
            return i + 15
        opt1 = self.parser.get_option('-v')
        self.assertIsInstance(opt1, Option)
        self.assertEqual(opt1._short_opts, ['-v', '-n'])
        self.assertEqual(opt1._long_opts, ['--verbose', '--noisy'])
        self.assertEqual(opt1.action, 'store_true')
        self.assertEqual(opt1.dest, 'verbose')

    def test_get_option_equals(self):
        if False:
            while True:
                i = 10
        opt1 = self.parser.get_option('-v')
        opt2 = self.parser.get_option('--verbose')
        opt3 = self.parser.get_option('-n')
        opt4 = self.parser.get_option('--noisy')
        self.assertTrue(opt1 is opt2 is opt3 is opt4)

    def test_has_option(self):
        if False:
            return 10
        self.assertTrue(self.parser.has_option('-v'))
        self.assertTrue(self.parser.has_option('--verbose'))

    def assertTrueremoved(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.parser.get_option('-v') is None)
        self.assertTrue(self.parser.get_option('--verbose') is None)
        self.assertTrue(self.parser.get_option('-n') is None)
        self.assertTrue(self.parser.get_option('--noisy') is None)
        self.assertFalse(self.parser.has_option('-v'))
        self.assertFalse(self.parser.has_option('--verbose'))
        self.assertFalse(self.parser.has_option('-n'))
        self.assertFalse(self.parser.has_option('--noisy'))
        self.assertTrue(self.parser.has_option('-q'))
        self.assertTrue(self.parser.has_option('--silent'))

    def test_remove_short_opt(self):
        if False:
            print('Hello World!')
        self.parser.remove_option('-n')
        self.assertTrueremoved()

    def test_remove_long_opt(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.remove_option('--verbose')
        self.assertTrueremoved()

    def test_remove_nonexistent(self):
        if False:
            while True:
                i = 10
        self.assertRaises(self.parser.remove_option, ('foo',), None, ValueError, "no such option 'foo'")

    @support.impl_detail('Relies on sys.getrefcount', cpython=True)
    def test_refleak(self):
        if False:
            i = 10
            return i + 15
        big_thing = [42]
        refcount = sys.getrefcount(big_thing)
        parser = OptionParser()
        parser.add_option('-a', '--aaarggh')
        parser.big_thing = big_thing
        parser.destroy()
        del parser
        self.assertEqual(refcount, sys.getrefcount(big_thing))

class TestOptionValues(BaseTest):

    def setUp(self):
        if False:
            return 10
        pass

    def test_basics(self):
        if False:
            print('Hello World!')
        values = Values()
        self.assertEqual(vars(values), {})
        self.assertEqual(values, {})
        self.assertNotEqual(values, {'foo': 'bar'})
        self.assertNotEqual(values, '')
        dict = {'foo': 'bar', 'baz': 42}
        values = Values(defaults=dict)
        self.assertEqual(vars(values), dict)
        self.assertEqual(values, dict)
        self.assertNotEqual(values, {'foo': 'bar'})
        self.assertNotEqual(values, {})
        self.assertNotEqual(values, '')
        self.assertNotEqual(values, [])

class TestTypeAliases(BaseTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser = OptionParser()

    def test_str_aliases_string(self):
        if False:
            print('Hello World!')
        self.parser.add_option('-s', type='str')
        self.assertEqual(self.parser.get_option('-s').type, 'string')

    def test_type_object(self):
        if False:
            while True:
                i = 10
        self.parser.add_option('-s', type=str)
        self.assertEqual(self.parser.get_option('-s').type, 'string')
        self.parser.add_option('-x', type=int)
        self.assertEqual(self.parser.get_option('-x').type, 'int')
_time_units = {'s': 1, 'm': 60, 'h': 60 * 60, 'd': 60 * 60 * 24}

def _check_duration(option, opt, value):
    if False:
        i = 10
        return i + 15
    try:
        if value[-1].isdigit():
            return int(value)
        else:
            return int(value[:-1]) * _time_units[value[-1]]
    except (ValueError, IndexError):
        raise OptionValueError('option %s: invalid duration: %r' % (opt, value))

class DurationOption(Option):
    TYPES = Option.TYPES + ('duration',)
    TYPE_CHECKER = copy.copy(Option.TYPE_CHECKER)
    TYPE_CHECKER['duration'] = _check_duration

class TestDefaultValues(BaseTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.parser = OptionParser()
        self.parser.add_option('-v', '--verbose', default=True)
        self.parser.add_option('-q', '--quiet', dest='verbose')
        self.parser.add_option('-n', type='int', default=37)
        self.parser.add_option('-m', type='int')
        self.parser.add_option('-s', default='foo')
        self.parser.add_option('-t')
        self.parser.add_option('-u', default=None)
        self.expected = {'verbose': True, 'n': 37, 'm': None, 's': 'foo', 't': None, 'u': None}

    def test_basic_defaults(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.parser.get_default_values(), self.expected)

    def test_mixed_defaults_post(self):
        if False:
            i = 10
            return i + 15
        self.parser.set_defaults(n=42, m=-100)
        self.expected.update({'n': 42, 'm': -100})
        self.assertEqual(self.parser.get_default_values(), self.expected)

    def test_mixed_defaults_pre(self):
        if False:
            while True:
                i = 10
        self.parser.set_defaults(x='barf', y='blah')
        self.parser.add_option('-x', default='frob')
        self.parser.add_option('-y')
        self.expected.update({'x': 'frob', 'y': 'blah'})
        self.assertEqual(self.parser.get_default_values(), self.expected)
        self.parser.remove_option('-y')
        self.parser.add_option('-y', default=None)
        self.expected.update({'y': None})
        self.assertEqual(self.parser.get_default_values(), self.expected)

    def test_process_default(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.option_class = DurationOption
        self.parser.add_option('-d', type='duration', default=300)
        self.parser.add_option('-e', type='duration', default='6m')
        self.parser.set_defaults(n='42')
        self.expected.update({'d': 300, 'e': 360, 'n': 42})
        self.assertEqual(self.parser.get_default_values(), self.expected)
        self.parser.set_process_default_values(False)
        self.expected.update({'d': 300, 'e': '6m', 'n': '42'})
        self.assertEqual(self.parser.get_default_values(), self.expected)

class TestProgName(BaseTest):
    """
    Test that %prog expands to the right thing in usage, version,
    and help strings.
    """

    def assertUsage(self, parser, expected_usage):
        if False:
            while True:
                i = 10
        self.assertEqual(parser.get_usage(), expected_usage)

    def assertVersion(self, parser, expected_version):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(parser.get_version(), expected_version)

    def test_default_progname(self):
        if False:
            while True:
                i = 10
        save_argv = sys.argv[:]
        try:
            sys.argv[0] = os.path.join('foo', 'bar', 'baz.py')
            parser = OptionParser('%prog ...', version='%prog 1.2')
            expected_usage = 'Usage: baz.py ...\n'
            self.assertUsage(parser, expected_usage)
            self.assertVersion(parser, 'baz.py 1.2')
            self.assertHelp(parser, expected_usage + '\n' + "Options:\n  --version   show program's version number and exit\n  -h, --help  show this help message and exit\n")
        finally:
            sys.argv[:] = save_argv

    def test_custom_progname(self):
        if False:
            for i in range(10):
                print('nop')
        parser = OptionParser(prog='thingy', version='%prog 0.1', usage='%prog arg arg')
        parser.remove_option('-h')
        parser.remove_option('--version')
        expected_usage = 'Usage: thingy arg arg\n'
        self.assertUsage(parser, expected_usage)
        self.assertVersion(parser, 'thingy 0.1')
        self.assertHelp(parser, expected_usage + '\n')

class TestExpandDefaults(BaseTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser = OptionParser(prog='test')
        self.help_prefix = 'Usage: test [options]\n\nOptions:\n  -h, --help            show this help message and exit\n'
        self.file_help = 'read from FILE [default: %default]'
        self.expected_help_file = self.help_prefix + '  -f FILE, --file=FILE  read from FILE [default: foo.txt]\n'
        self.expected_help_none = self.help_prefix + '  -f FILE, --file=FILE  read from FILE [default: none]\n'

    def test_option_default(self):
        if False:
            return 10
        self.parser.add_option('-f', '--file', default='foo.txt', help=self.file_help)
        self.assertHelp(self.parser, self.expected_help_file)

    def test_parser_default_1(self):
        if False:
            while True:
                i = 10
        self.parser.add_option('-f', '--file', help=self.file_help)
        self.parser.set_default('file', 'foo.txt')
        self.assertHelp(self.parser, self.expected_help_file)

    def test_parser_default_2(self):
        if False:
            i = 10
            return i + 15
        self.parser.add_option('-f', '--file', help=self.file_help)
        self.parser.set_defaults(file='foo.txt')
        self.assertHelp(self.parser, self.expected_help_file)

    def test_no_default(self):
        if False:
            print('Hello World!')
        self.parser.add_option('-f', '--file', help=self.file_help)
        self.assertHelp(self.parser, self.expected_help_none)

    def test_default_none_1(self):
        if False:
            print('Hello World!')
        self.parser.add_option('-f', '--file', default=None, help=self.file_help)
        self.assertHelp(self.parser, self.expected_help_none)

    def test_default_none_2(self):
        if False:
            i = 10
            return i + 15
        self.parser.add_option('-f', '--file', help=self.file_help)
        self.parser.set_defaults(file=None)
        self.assertHelp(self.parser, self.expected_help_none)

    def test_float_default(self):
        if False:
            print('Hello World!')
        self.parser.add_option('-p', '--prob', help='blow up with probability PROB [default: %default]')
        self.parser.set_defaults(prob=0.43)
        expected_help = self.help_prefix + '  -p PROB, --prob=PROB  blow up with probability PROB [default: 0.43]\n'
        self.assertHelp(self.parser, expected_help)

    def test_alt_expand(self):
        if False:
            print('Hello World!')
        self.parser.add_option('-f', '--file', default='foo.txt', help='read from FILE [default: *DEFAULT*]')
        self.parser.formatter.default_tag = '*DEFAULT*'
        self.assertHelp(self.parser, self.expected_help_file)

    def test_no_expand(self):
        if False:
            print('Hello World!')
        self.parser.add_option('-f', '--file', default='foo.txt', help='read from %default file')
        self.parser.formatter.default_tag = None
        expected_help = self.help_prefix + '  -f FILE, --file=FILE  read from %default file\n'
        self.assertHelp(self.parser, expected_help)

class TestStandard(BaseTest):

    def setUp(self):
        if False:
            return 10
        options = [make_option('-a', type='string'), make_option('-b', '--boo', type='int', dest='boo'), make_option('--foo', action='append')]
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE, option_list=options)

    def test_required_value(self):
        if False:
            while True:
                i = 10
        self.assertParseFail(['-a'], '-a option requires 1 argument')

    def test_invalid_integer(self):
        if False:
            print('Hello World!')
        self.assertParseFail(['-b', '5x'], "option -b: invalid integer value: '5x'")

    def test_no_such_option(self):
        if False:
            i = 10
            return i + 15
        self.assertParseFail(['--boo13'], 'no such option: --boo13')

    def test_long_invalid_integer(self):
        if False:
            while True:
                i = 10
        self.assertParseFail(['--boo=x5'], "option --boo: invalid integer value: 'x5'")

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK([], {'a': None, 'boo': None, 'foo': None}, [])

    def test_shortopt_empty_longopt_append(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK(['-a', '', '--foo=blah', '--foo='], {'a': '', 'boo': None, 'foo': ['blah', '']}, [])

    def test_long_option_append(self):
        if False:
            return 10
        self.assertParseOK(['--foo', 'bar', '--foo', '', '--foo=x'], {'a': None, 'boo': None, 'foo': ['bar', '', 'x']}, [])

    def test_option_argument_joined(self):
        if False:
            print('Hello World!')
        self.assertParseOK(['-abc'], {'a': 'bc', 'boo': None, 'foo': None}, [])

    def test_option_argument_split(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK(['-a', '34'], {'a': '34', 'boo': None, 'foo': None}, [])

    def test_option_argument_joined_integer(self):
        if False:
            return 10
        self.assertParseOK(['-b34'], {'a': None, 'boo': 34, 'foo': None}, [])

    def test_option_argument_split_negative_integer(self):
        if False:
            print('Hello World!')
        self.assertParseOK(['-b', '-5'], {'a': None, 'boo': -5, 'foo': None}, [])

    def test_long_option_argument_joined(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK(['--boo=13'], {'a': None, 'boo': 13, 'foo': None}, [])

    def test_long_option_argument_split(self):
        if False:
            while True:
                i = 10
        self.assertParseOK(['--boo', '111'], {'a': None, 'boo': 111, 'foo': None}, [])

    def test_long_option_short_option(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK(['--foo=bar', '-axyz'], {'a': 'xyz', 'boo': None, 'foo': ['bar']}, [])

    def test_abbrev_long_option(self):
        if False:
            return 10
        self.assertParseOK(['--f=bar', '-axyz'], {'a': 'xyz', 'boo': None, 'foo': ['bar']}, [])

    def test_defaults(self):
        if False:
            i = 10
            return i + 15
        (options, args) = self.parser.parse_args([])
        defaults = self.parser.get_default_values()
        self.assertEqual(vars(defaults), vars(options))

    def test_ambiguous_option(self):
        if False:
            i = 10
            return i + 15
        self.parser.add_option('--foz', action='store', type='string', dest='foo')
        self.assertParseFail(['--f=bar'], 'ambiguous option: --f (--foo, --foz?)')

    def test_short_and_long_option_split(self):
        if False:
            print('Hello World!')
        self.assertParseOK(['-a', 'xyz', '--foo', 'bar'], {'a': 'xyz', 'boo': None, 'foo': ['bar']}, [])

    def test_short_option_split_long_option_append(self):
        if False:
            print('Hello World!')
        self.assertParseOK(['--foo=bar', '-b', '123', '--foo', 'baz'], {'a': None, 'boo': 123, 'foo': ['bar', 'baz']}, [])

    def test_short_option_split_one_positional_arg(self):
        if False:
            while True:
                i = 10
        self.assertParseOK(['-a', 'foo', 'bar'], {'a': 'foo', 'boo': None, 'foo': None}, ['bar'])

    def test_short_option_consumes_separator(self):
        if False:
            return 10
        self.assertParseOK(['-a', '--', 'foo', 'bar'], {'a': '--', 'boo': None, 'foo': None}, ['foo', 'bar'])
        self.assertParseOK(['-a', '--', '--foo', 'bar'], {'a': '--', 'boo': None, 'foo': ['bar']}, [])

    def test_short_option_joined_and_separator(self):
        if False:
            for i in range(10):
                print('nop')
        (self.assertParseOK(['-ab', '--', '--foo', 'bar'], {'a': 'b', 'boo': None, 'foo': None}, ['--foo', 'bar']),)

    def test_hyphen_becomes_positional_arg(self):
        if False:
            return 10
        self.assertParseOK(['-ab', '-', '--foo', 'bar'], {'a': 'b', 'boo': None, 'foo': ['bar']}, ['-'])

    def test_no_append_versus_append(self):
        if False:
            print('Hello World!')
        self.assertParseOK(['-b3', '-b', '5', '--foo=bar', '--foo', 'baz'], {'a': None, 'boo': 5, 'foo': ['bar', 'baz']}, [])

    def test_option_consumes_optionlike_string(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK(['-a', '-b3'], {'a': '-b3', 'boo': None, 'foo': None}, [])

    def test_combined_single_invalid_option(self):
        if False:
            i = 10
            return i + 15
        self.parser.add_option('-t', action='store_true')
        self.assertParseFail(['-test'], 'no such option: -e')

class TestBool(BaseTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        options = [make_option('-v', '--verbose', action='store_true', dest='verbose', default=''), make_option('-q', '--quiet', action='store_false', dest='verbose')]
        self.parser = OptionParser(option_list=options)

    def test_bool_default(self):
        if False:
            return 10
        self.assertParseOK([], {'verbose': ''}, [])

    def test_bool_false(self):
        if False:
            print('Hello World!')
        (options, args) = self.assertParseOK(['-q'], {'verbose': 0}, [])
        self.assertTrue(options.verbose is False)

    def test_bool_true(self):
        if False:
            print('Hello World!')
        (options, args) = self.assertParseOK(['-v'], {'verbose': 1}, [])
        self.assertTrue(options.verbose is True)

    def test_bool_flicker_on_and_off(self):
        if False:
            return 10
        self.assertParseOK(['-qvq', '-q', '-v'], {'verbose': 1}, [])

class TestChoice(BaseTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE)
        self.parser.add_option('-c', action='store', type='choice', dest='choice', choices=['one', 'two', 'three'])

    def test_valid_choice(self):
        if False:
            while True:
                i = 10
        self.assertParseOK(['-c', 'one', 'xyz'], {'choice': 'one'}, ['xyz'])

    def test_invalid_choice(self):
        if False:
            i = 10
            return i + 15
        self.assertParseFail(['-c', 'four', 'abc'], "option -c: invalid choice: 'four' (choose from 'one', 'two', 'three')")

    def test_add_choice_option(self):
        if False:
            print('Hello World!')
        self.parser.add_option('-d', '--default', choices=['four', 'five', 'six'])
        opt = self.parser.get_option('-d')
        self.assertEqual(opt.type, 'choice')
        self.assertEqual(opt.action, 'store')

class TestCount(BaseTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE)
        self.v_opt = make_option('-v', action='count', dest='verbose')
        self.parser.add_option(self.v_opt)
        self.parser.add_option('--verbose', type='int', dest='verbose')
        self.parser.add_option('-q', '--quiet', action='store_const', dest='verbose', const=0)

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK([], {'verbose': None}, [])

    def test_count_one(self):
        if False:
            print('Hello World!')
        self.assertParseOK(['-v'], {'verbose': 1}, [])

    def test_count_three(self):
        if False:
            while True:
                i = 10
        self.assertParseOK(['-vvv'], {'verbose': 3}, [])

    def test_count_three_apart(self):
        if False:
            i = 10
            return i + 15
        self.assertParseOK(['-v', '-v', '-v'], {'verbose': 3}, [])

    def test_count_override_amount(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK(['-vvv', '--verbose=2'], {'verbose': 2}, [])

    def test_count_override_quiet(self):
        if False:
            while True:
                i = 10
        self.assertParseOK(['-vvv', '--verbose=2', '-q'], {'verbose': 0}, [])

    def test_count_overriding(self):
        if False:
            i = 10
            return i + 15
        self.assertParseOK(['-vvv', '--verbose=2', '-q', '-v'], {'verbose': 1}, [])

    def test_count_interspersed_args(self):
        if False:
            i = 10
            return i + 15
        self.assertParseOK(['--quiet', '3', '-v'], {'verbose': 1}, ['3'])

    def test_count_no_interspersed_args(self):
        if False:
            while True:
                i = 10
        self.parser.disable_interspersed_args()
        self.assertParseOK(['--quiet', '3', '-v'], {'verbose': 0}, ['3', '-v'])

    def test_count_no_such_option(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseFail(['-q3', '-v'], 'no such option: -3')

    def test_count_option_no_value(self):
        if False:
            print('Hello World!')
        self.assertParseFail(['--quiet=3', '-v'], '--quiet option does not take a value')

    def test_count_with_default(self):
        if False:
            while True:
                i = 10
        self.parser.set_default('verbose', 0)
        self.assertParseOK([], {'verbose': 0}, [])

    def test_count_overriding_default(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.set_default('verbose', 0)
        self.assertParseOK(['-vvv', '--verbose=2', '-q', '-v'], {'verbose': 1}, [])

class TestMultipleArgs(BaseTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE)
        self.parser.add_option('-p', '--point', action='store', nargs=3, type='float', dest='point')

    def test_nargs_with_positional_args(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK(['foo', '-p', '1', '2.5', '-4.3', 'xyz'], {'point': (1.0, 2.5, -4.3)}, ['foo', 'xyz'])

    def test_nargs_long_opt(self):
        if False:
            while True:
                i = 10
        self.assertParseOK(['--point', '-1', '2.5', '-0', 'xyz'], {'point': (-1.0, 2.5, -0.0)}, ['xyz'])

    def test_nargs_invalid_float_value(self):
        if False:
            print('Hello World!')
        self.assertParseFail(['-p', '1.0', '2x', '3.5'], "option -p: invalid floating-point value: '2x'")

    def test_nargs_required_values(self):
        if False:
            return 10
        self.assertParseFail(['--point', '1.0', '3.5'], '--point option requires 3 arguments')

class TestMultipleArgsAppend(BaseTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE)
        self.parser.add_option('-p', '--point', action='store', nargs=3, type='float', dest='point')
        self.parser.add_option('-f', '--foo', action='append', nargs=2, type='int', dest='foo')
        self.parser.add_option('-z', '--zero', action='append_const', dest='foo', const=(0, 0))

    def test_nargs_append(self):
        if False:
            return 10
        self.assertParseOK(['-f', '4', '-3', 'blah', '--foo', '1', '666'], {'point': None, 'foo': [(4, -3), (1, 666)]}, ['blah'])

    def test_nargs_append_required_values(self):
        if False:
            print('Hello World!')
        self.assertParseFail(['-f4,3'], '-f option requires 2 arguments')

    def test_nargs_append_simple(self):
        if False:
            print('Hello World!')
        self.assertParseOK(['--foo=3', '4'], {'point': None, 'foo': [(3, 4)]}, [])

    def test_nargs_append_const(self):
        if False:
            while True:
                i = 10
        self.assertParseOK(['--zero', '--foo', '3', '4', '-z'], {'point': None, 'foo': [(0, 0), (3, 4), (0, 0)]}, [])

class TestVersion(BaseTest):

    def test_version(self):
        if False:
            i = 10
            return i + 15
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE, version='%prog 0.1')
        save_argv = sys.argv[:]
        try:
            sys.argv[0] = os.path.join(os.curdir, 'foo', 'bar')
            self.assertOutput(['--version'], 'bar 0.1\n')
        finally:
            sys.argv[:] = save_argv

    def test_no_version(self):
        if False:
            while True:
                i = 10
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE)
        self.assertParseFail(['--version'], 'no such option: --version')

class TestConflictingDefaults(BaseTest):
    """Conflicting default values: the last one should win."""

    def setUp(self):
        if False:
            return 10
        self.parser = OptionParser(option_list=[make_option('-v', action='store_true', dest='verbose', default=1)])

    def test_conflict_default(self):
        if False:
            while True:
                i = 10
        self.parser.add_option('-q', action='store_false', dest='verbose', default=0)
        self.assertParseOK([], {'verbose': 0}, [])

    def test_conflict_default_none(self):
        if False:
            while True:
                i = 10
        self.parser.add_option('-q', action='store_false', dest='verbose', default=None)
        self.assertParseOK([], {'verbose': None}, [])

class TestOptionGroup(BaseTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.parser = OptionParser(usage=SUPPRESS_USAGE)

    def test_option_group_create_instance(self):
        if False:
            while True:
                i = 10
        group = OptionGroup(self.parser, 'Spam')
        self.parser.add_option_group(group)
        group.add_option('--spam', action='store_true', help='spam spam spam spam')
        self.assertParseOK(['--spam'], {'spam': 1}, [])

    def test_add_group_no_group(self):
        if False:
            while True:
                i = 10
        self.assertTypeError(self.parser.add_option_group, 'not an OptionGroup instance: None', None)

    def test_add_group_invalid_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTypeError(self.parser.add_option_group, 'invalid arguments', None, None)

    def test_add_group_wrong_parser(self):
        if False:
            for i in range(10):
                print('nop')
        group = OptionGroup(self.parser, 'Spam')
        group.parser = OptionParser()
        self.assertRaises(self.parser.add_option_group, (group,), None, ValueError, 'invalid OptionGroup (wrong parser)')

    def test_group_manipulate(self):
        if False:
            print('Hello World!')
        group = self.parser.add_option_group('Group 2', description='Some more options')
        group.set_title('Bacon')
        group.add_option('--bacon', type='int')
        self.assertTrue(self.parser.get_option_group('--bacon'), group)

class TestExtendAddTypes(BaseTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE, option_class=self.MyOption)
        self.parser.add_option('-a', None, type='string', dest='a')
        self.parser.add_option('-f', '--file', type='file', dest='file')

    def tearDown(self):
        if False:
            return 10
        if os.path.isdir(os_helper.TESTFN):
            os.rmdir(os_helper.TESTFN)
        elif os.path.isfile(os_helper.TESTFN):
            os.unlink(os_helper.TESTFN)

    class MyOption(Option):

        def check_file(option, opt, value):
            if False:
                while True:
                    i = 10
            if not os.path.exists(value):
                raise OptionValueError('%s: file does not exist' % value)
            elif not os.path.isfile(value):
                raise OptionValueError('%s: not a regular file' % value)
            return value
        TYPES = Option.TYPES + ('file',)
        TYPE_CHECKER = copy.copy(Option.TYPE_CHECKER)
        TYPE_CHECKER['file'] = check_file

    def test_filetype_ok(self):
        if False:
            i = 10
            return i + 15
        os_helper.create_empty_file(os_helper.TESTFN)
        self.assertParseOK(['--file', os_helper.TESTFN, '-afoo'], {'file': os_helper.TESTFN, 'a': 'foo'}, [])

    def test_filetype_noexist(self):
        if False:
            print('Hello World!')
        self.assertParseFail(['--file', os_helper.TESTFN, '-afoo'], '%s: file does not exist' % os_helper.TESTFN)

    def test_filetype_notfile(self):
        if False:
            print('Hello World!')
        os.mkdir(os_helper.TESTFN)
        self.assertParseFail(['--file', os_helper.TESTFN, '-afoo'], '%s: not a regular file' % os_helper.TESTFN)

class TestExtendAddActions(BaseTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        options = [self.MyOption('-a', '--apple', action='extend', type='string', dest='apple')]
        self.parser = OptionParser(option_list=options)

    class MyOption(Option):
        ACTIONS = Option.ACTIONS + ('extend',)
        STORE_ACTIONS = Option.STORE_ACTIONS + ('extend',)
        TYPED_ACTIONS = Option.TYPED_ACTIONS + ('extend',)

        def take_action(self, action, dest, opt, value, values, parser):
            if False:
                while True:
                    i = 10
            if action == 'extend':
                lvalue = value.split(',')
                values.ensure_value(dest, []).extend(lvalue)
            else:
                Option.take_action(self, action, dest, opt, parser, value, values)

    def test_extend_add_action(self):
        if False:
            return 10
        self.assertParseOK(['-afoo,bar', '--apple=blah'], {'apple': ['foo', 'bar', 'blah']}, [])

    def test_extend_add_action_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK(['-a', 'foo', '-abar', '--apple=x,y'], {'apple': ['foo', 'bar', 'x', 'y']}, [])

class TestCallback(BaseTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        options = [make_option('-x', None, action='callback', callback=self.process_opt), make_option('-f', '--file', action='callback', callback=self.process_opt, type='string', dest='filename')]
        self.parser = OptionParser(option_list=options)

    def process_opt(self, option, opt, value, parser_):
        if False:
            return 10
        if opt == '-x':
            self.assertEqual(option._short_opts, ['-x'])
            self.assertEqual(option._long_opts, [])
            self.assertTrue(parser_ is self.parser)
            self.assertTrue(value is None)
            self.assertEqual(vars(parser_.values), {'filename': None})
            parser_.values.x = 42
        elif opt == '--file':
            self.assertEqual(option._short_opts, ['-f'])
            self.assertEqual(option._long_opts, ['--file'])
            self.assertTrue(parser_ is self.parser)
            self.assertEqual(value, 'foo')
            self.assertEqual(vars(parser_.values), {'filename': None, 'x': 42})
            setattr(parser_.values, option.dest, value)
        else:
            self.fail('Unknown option %r in process_opt.' % opt)

    def test_callback(self):
        if False:
            print('Hello World!')
        self.assertParseOK(['-x', '--file=foo'], {'filename': 'foo', 'x': 42}, [])

    def test_callback_help(self):
        if False:
            i = 10
            return i + 15
        parser = OptionParser(usage=SUPPRESS_USAGE)
        parser.remove_option('-h')
        parser.add_option('-t', '--test', action='callback', callback=lambda : None, type='string', help='foo')
        expected_help = 'Options:\n  -t TEST, --test=TEST  foo\n'
        self.assertHelp(parser, expected_help)

class TestCallbackExtraArgs(BaseTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        options = [make_option('-p', '--point', action='callback', callback=self.process_tuple, callback_args=(3, int), type='string', dest='points', default=[])]
        self.parser = OptionParser(option_list=options)

    def process_tuple(self, option, opt, value, parser_, len, type):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len, 3)
        self.assertTrue(type is int)
        if opt == '-p':
            self.assertEqual(value, '1,2,3')
        elif opt == '--point':
            self.assertEqual(value, '4,5,6')
        value = tuple(map(type, value.split(',')))
        getattr(parser_.values, option.dest).append(value)

    def test_callback_extra_args(self):
        if False:
            return 10
        self.assertParseOK(['-p1,2,3', '--point', '4,5,6'], {'points': [(1, 2, 3), (4, 5, 6)]}, [])

class TestCallbackMeddleArgs(BaseTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        options = [make_option(str(x), action='callback', callback=self.process_n, dest='things') for x in range(-1, -6, -1)]
        self.parser = OptionParser(option_list=options)

    def process_n(self, option, opt, value, parser_):
        if False:
            while True:
                i = 10
        nargs = int(opt[1:])
        rargs = parser_.rargs
        if len(rargs) < nargs:
            self.fail('Expected %d arguments for %s option.' % (nargs, opt))
        dest = parser_.values.ensure_value(option.dest, [])
        dest.append(tuple(rargs[0:nargs]))
        parser_.largs.append(nargs)
        del rargs[0:nargs]

    def test_callback_meddle_args(self):
        if False:
            return 10
        self.assertParseOK(['-1', 'foo', '-3', 'bar', 'baz', 'qux'], {'things': [('foo',), ('bar', 'baz', 'qux')]}, [1, 3])

    def test_callback_meddle_args_separator(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK(['-2', 'foo', '--'], {'things': [('foo', '--')]}, [2])

class TestCallbackManyArgs(BaseTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        options = [make_option('-a', '--apple', action='callback', nargs=2, callback=self.process_many, type='string'), make_option('-b', '--bob', action='callback', nargs=3, callback=self.process_many, type='int')]
        self.parser = OptionParser(option_list=options)

    def process_many(self, option, opt, value, parser_):
        if False:
            i = 10
            return i + 15
        if opt == '-a':
            self.assertEqual(value, ('foo', 'bar'))
        elif opt == '--apple':
            self.assertEqual(value, ('ding', 'dong'))
        elif opt == '-b':
            self.assertEqual(value, (1, 2, 3))
        elif opt == '--bob':
            self.assertEqual(value, (-666, 42, 0))

    def test_many_args(self):
        if False:
            while True:
                i = 10
        self.assertParseOK(['-a', 'foo', 'bar', '--apple', 'ding', 'dong', '-b', '1', '2', '3', '--bob', '-666', '42', '0'], {'apple': None, 'bob': None}, [])

class TestCallbackCheckAbbrev(BaseTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.parser = OptionParser()
        self.parser.add_option('--foo-bar', action='callback', callback=self.check_abbrev)

    def check_abbrev(self, option, opt, value, parser):
        if False:
            print('Hello World!')
        self.assertEqual(opt, '--foo-bar')

    def test_abbrev_callback_expansion(self):
        if False:
            print('Hello World!')
        self.assertParseOK(['--foo'], {}, [])

class TestCallbackVarArgs(BaseTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        options = [make_option('-a', type='int', nargs=2, dest='a'), make_option('-b', action='store_true', dest='b'), make_option('-c', '--callback', action='callback', callback=self.variable_args, dest='c')]
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE, option_list=options)

    def variable_args(self, option, opt, value, parser):
        if False:
            while True:
                i = 10
        self.assertTrue(value is None)
        value = []
        rargs = parser.rargs
        while rargs:
            arg = rargs[0]
            if arg[:2] == '--' and len(arg) > 2 or (arg[:1] == '-' and len(arg) > 1 and (arg[1] != '-')):
                break
            else:
                value.append(arg)
                del rargs[0]
        setattr(parser.values, option.dest, value)

    def test_variable_args(self):
        if False:
            print('Hello World!')
        self.assertParseOK(['-a3', '-5', '--callback', 'foo', 'bar'], {'a': (3, -5), 'b': None, 'c': ['foo', 'bar']}, [])

    def test_consume_separator_stop_at_option(self):
        if False:
            i = 10
            return i + 15
        self.assertParseOK(['-c', '37', '--', 'xxx', '-b', 'hello'], {'a': None, 'b': True, 'c': ['37', '--', 'xxx']}, ['hello'])

    def test_positional_arg_and_variable_args(self):
        if False:
            while True:
                i = 10
        self.assertParseOK(['hello', '-c', 'foo', '-', 'bar'], {'a': None, 'b': None, 'c': ['foo', '-', 'bar']}, ['hello'])

    def test_stop_at_option(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertParseOK(['-c', 'foo', '-b'], {'a': None, 'b': True, 'c': ['foo']}, [])

    def test_stop_at_invalid_option(self):
        if False:
            while True:
                i = 10
        self.assertParseFail(['-c', '3', '-5', '-a'], 'no such option: -5')

class ConflictBase(BaseTest):

    def setUp(self):
        if False:
            return 10
        options = [make_option('-v', '--verbose', action='count', dest='verbose', help='increment verbosity')]
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE, option_list=options)

    def show_version(self, option, opt, value, parser):
        if False:
            i = 10
            return i + 15
        parser.values.show_version = 1

class TestConflict(ConflictBase):
    """Use the default conflict resolution for Optik 1.2: error."""

    def assertTrueconflict_error(self, func):
        if False:
            return 10
        err = self.assertRaises(func, ('-v', '--version'), {'action': 'callback', 'callback': self.show_version, 'help': 'show version'}, OptionConflictError, 'option -v/--version: conflicting option string(s): -v')
        self.assertEqual(err.msg, 'conflicting option string(s): -v')
        self.assertEqual(err.option_id, '-v/--version')

    def test_conflict_error(self):
        if False:
            i = 10
            return i + 15
        self.assertTrueconflict_error(self.parser.add_option)

    def test_conflict_error_group(self):
        if False:
            i = 10
            return i + 15
        group = OptionGroup(self.parser, 'Group 1')
        self.assertTrueconflict_error(group.add_option)

    def test_no_such_conflict_handler(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(self.parser.set_conflict_handler, ('foo',), None, ValueError, "invalid conflict_resolution value 'foo'")

class TestConflictResolve(ConflictBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        ConflictBase.setUp(self)
        self.parser.set_conflict_handler('resolve')
        self.parser.add_option('-v', '--version', action='callback', callback=self.show_version, help='show version')

    def test_conflict_resolve(self):
        if False:
            return 10
        v_opt = self.parser.get_option('-v')
        verbose_opt = self.parser.get_option('--verbose')
        version_opt = self.parser.get_option('--version')
        self.assertTrue(v_opt is version_opt)
        self.assertTrue(v_opt is not verbose_opt)
        self.assertEqual(v_opt._long_opts, ['--version'])
        self.assertEqual(version_opt._short_opts, ['-v'])
        self.assertEqual(version_opt._long_opts, ['--version'])
        self.assertEqual(verbose_opt._short_opts, [])
        self.assertEqual(verbose_opt._long_opts, ['--verbose'])

    def test_conflict_resolve_help(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertOutput(['-h'], 'Options:\n  --verbose      increment verbosity\n  -h, --help     show this help message and exit\n  -v, --version  show version\n')

    def test_conflict_resolve_short_opt(self):
        if False:
            i = 10
            return i + 15
        self.assertParseOK(['-v'], {'verbose': None, 'show_version': 1}, [])

    def test_conflict_resolve_long_opt(self):
        if False:
            while True:
                i = 10
        self.assertParseOK(['--verbose'], {'verbose': 1}, [])

    def test_conflict_resolve_long_opts(self):
        if False:
            while True:
                i = 10
        self.assertParseOK(['--verbose', '--version'], {'verbose': 1, 'show_version': 1}, [])

class TestConflictOverride(BaseTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE)
        self.parser.set_conflict_handler('resolve')
        self.parser.add_option('-n', '--dry-run', action='store_true', dest='dry_run', help="don't do anything")
        self.parser.add_option('--dry-run', '-n', action='store_const', const=42, dest='dry_run', help='dry run mode')

    def test_conflict_override_opts(self):
        if False:
            return 10
        opt = self.parser.get_option('--dry-run')
        self.assertEqual(opt._short_opts, ['-n'])
        self.assertEqual(opt._long_opts, ['--dry-run'])

    def test_conflict_override_help(self):
        if False:
            while True:
                i = 10
        self.assertOutput(['-h'], 'Options:\n  -h, --help     show this help message and exit\n  -n, --dry-run  dry run mode\n')

    def test_conflict_override_args(self):
        if False:
            i = 10
            return i + 15
        self.assertParseOK(['-n'], {'dry_run': 42}, [])
_expected_help_basic = 'Usage: bar.py [options]\n\nOptions:\n  -a APPLE           throw APPLEs at basket\n  -b NUM, --boo=NUM  shout "boo!" NUM times (in order to frighten away all the\n                     evil spirits that cause trouble and mayhem)\n  --foo=FOO          store FOO in the foo list for later fooing\n  -h, --help         show this help message and exit\n'
_expected_help_long_opts_first = 'Usage: bar.py [options]\n\nOptions:\n  -a APPLE           throw APPLEs at basket\n  --boo=NUM, -b NUM  shout "boo!" NUM times (in order to frighten away all the\n                     evil spirits that cause trouble and mayhem)\n  --foo=FOO          store FOO in the foo list for later fooing\n  --help, -h         show this help message and exit\n'
_expected_help_title_formatter = 'Usage\n=====\n  bar.py [options]\n\nOptions\n=======\n-a APPLE           throw APPLEs at basket\n--boo=NUM, -b NUM  shout "boo!" NUM times (in order to frighten away all the\n                   evil spirits that cause trouble and mayhem)\n--foo=FOO          store FOO in the foo list for later fooing\n--help, -h         show this help message and exit\n'
_expected_help_short_lines = 'Usage: bar.py [options]\n\nOptions:\n  -a APPLE           throw APPLEs at basket\n  -b NUM, --boo=NUM  shout "boo!" NUM times (in order to\n                     frighten away all the evil spirits\n                     that cause trouble and mayhem)\n  --foo=FOO          store FOO in the foo list for later\n                     fooing\n  -h, --help         show this help message and exit\n'
_expected_very_help_short_lines = 'Usage: bar.py [options]\n\nOptions:\n  -a APPLE\n    throw\n    APPLEs at\n    basket\n  -b NUM, --boo=NUM\n    shout\n    "boo!" NUM\n    times (in\n    order to\n    frighten\n    away all\n    the evil\n    spirits\n    that cause\n    trouble and\n    mayhem)\n  --foo=FOO\n    store FOO\n    in the foo\n    list for\n    later\n    fooing\n  -h, --help\n    show this\n    help\n    message and\n    exit\n'

class TestHelp(BaseTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.parser = self.make_parser(80)

    def make_parser(self, columns):
        if False:
            print('Hello World!')
        options = [make_option('-a', type='string', dest='a', metavar='APPLE', help='throw APPLEs at basket'), make_option('-b', '--boo', type='int', dest='boo', metavar='NUM', help='shout "boo!" NUM times (in order to frighten away all the evil spirits that cause trouble and mayhem)'), make_option('--foo', action='append', type='string', dest='foo', help='store FOO in the foo list for later fooing')]
        with os_helper.EnvironmentVarGuard() as env:
            env['COLUMNS'] = str(columns)
            return InterceptingOptionParser(option_list=options)

    def assertHelpEquals(self, expected_output):
        if False:
            return 10
        save_argv = sys.argv[:]
        try:
            sys.argv[0] = os.path.join('foo', 'bar.py')
            self.assertOutput(['-h'], expected_output)
        finally:
            sys.argv[:] = save_argv

    def test_help(self):
        if False:
            print('Hello World!')
        self.assertHelpEquals(_expected_help_basic)

    def test_help_old_usage(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.set_usage('Usage: %prog [options]')
        self.assertHelpEquals(_expected_help_basic)

    def test_help_long_opts_first(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.formatter.short_first = 0
        self.assertHelpEquals(_expected_help_long_opts_first)

    def test_help_title_formatter(self):
        if False:
            while True:
                i = 10
        with os_helper.EnvironmentVarGuard() as env:
            env['COLUMNS'] = '80'
            self.parser.formatter = TitledHelpFormatter()
            self.assertHelpEquals(_expected_help_title_formatter)

    def test_wrap_columns(self):
        if False:
            print('Hello World!')
        self.parser = self.make_parser(60)
        self.assertHelpEquals(_expected_help_short_lines)
        self.parser = self.make_parser(0)
        self.assertHelpEquals(_expected_very_help_short_lines)

    def test_help_unicode(self):
        if False:
            print('Hello World!')
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE)
        self.parser.add_option('-a', action='store_true', help='olé!')
        expect = 'Options:\n  -h, --help  show this help message and exit\n  -a          olé!\n'
        self.assertHelpEquals(expect)

    def test_help_unicode_description(self):
        if False:
            i = 10
            return i + 15
        self.parser = InterceptingOptionParser(usage=SUPPRESS_USAGE, description='olé!')
        expect = 'olé!\n\nOptions:\n  -h, --help  show this help message and exit\n'
        self.assertHelpEquals(expect)

    def test_help_description_groups(self):
        if False:
            for i in range(10):
                print('nop')
        self.parser.set_description('This is the program description for %prog.  %prog has an option group as well as single options.')
        group = OptionGroup(self.parser, 'Dangerous Options', 'Caution: use of these options is at your own risk.  It is believed that some of them bite.')
        group.add_option('-g', action='store_true', help='Group option.')
        self.parser.add_option_group(group)
        expect = 'Usage: bar.py [options]\n\nThis is the program description for bar.py.  bar.py has an option group as\nwell as single options.\n\nOptions:\n  -a APPLE           throw APPLEs at basket\n  -b NUM, --boo=NUM  shout "boo!" NUM times (in order to frighten away all the\n                     evil spirits that cause trouble and mayhem)\n  --foo=FOO          store FOO in the foo list for later fooing\n  -h, --help         show this help message and exit\n\n  Dangerous Options:\n    Caution: use of these options is at your own risk.  It is believed\n    that some of them bite.\n\n    -g               Group option.\n'
        self.assertHelpEquals(expect)
        self.parser.epilog = 'Please report bugs to /dev/null.'
        self.assertHelpEquals(expect + '\nPlease report bugs to /dev/null.\n')

class TestMatchAbbrev(BaseTest):

    def test_match_abbrev(self):
        if False:
            return 10
        self.assertEqual(_match_abbrev('--f', {'--foz': None, '--foo': None, '--fie': None, '--f': None}), '--f')

    def test_match_abbrev_error(self):
        if False:
            while True:
                i = 10
        s = '--f'
        wordmap = {'--foz': None, '--foo': None, '--fie': None}
        self.assertRaises(_match_abbrev, (s, wordmap), None, BadOptionError, 'ambiguous option: --f (--fie, --foo, --foz?)')

class TestParseNumber(BaseTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.parser = InterceptingOptionParser()
        self.parser.add_option('-n', type=int)
        self.parser.add_option('-l', type=int)

    def test_parse_num_fail(self):
        if False:
            while True:
                i = 10
        self.assertRaises(_parse_num, ('', int), {}, ValueError, re.compile("invalid literal for int().*: '?'?"))
        self.assertRaises(_parse_num, ('0xOoops', int), {}, ValueError, re.compile("invalid literal for int().*: s?'?0xOoops'?"))

    def test_parse_num_ok(self):
        if False:
            while True:
                i = 10
        self.assertEqual(_parse_num('0', int), 0)
        self.assertEqual(_parse_num('0x10', int), 16)
        self.assertEqual(_parse_num('0XA', int), 10)
        self.assertEqual(_parse_num('010', int), 8)
        self.assertEqual(_parse_num('0b11', int), 3)
        self.assertEqual(_parse_num('0b', int), 0)

    def test_numeric_options(self):
        if False:
            return 10
        self.assertParseOK(['-n', '42', '-l', '0x20'], {'n': 42, 'l': 32}, [])
        self.assertParseOK(['-n', '0b0101', '-l010'], {'n': 5, 'l': 8}, [])
        self.assertParseFail(['-n008'], "option -n: invalid integer value: '008'")
        self.assertParseFail(['-l0b0123'], "option -l: invalid integer value: '0b0123'")
        self.assertParseFail(['-l', '0x12x'], "option -l: invalid integer value: '0x12x'")

class MiscTestCase(unittest.TestCase):

    def test__all__(self):
        if False:
            while True:
                i = 10
        not_exported = {'check_builtin', 'AmbiguousOptionError', 'NO_DEFAULT'}
        support.check__all__(self, optparse, not_exported=not_exported)
if __name__ == '__main__':
    unittest.main()