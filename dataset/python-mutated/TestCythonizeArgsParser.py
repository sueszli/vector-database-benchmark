from Cython.Build.Cythonize import create_args_parser, parse_args_raw, parse_args, parallel_compiles
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

class TestCythonizeArgsParser(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        TestCase.setUp(self)
        self.parse_args = lambda x, parser=create_args_parser(): parse_args_raw(parser, x)

    def are_default(self, options, skip):
        if False:
            while True:
                i = 10
        empty_containers = ['directives', 'compile_time_env', 'options', 'excludes']
        are_none = ['language_level', 'annotate', 'build', 'build_inplace', 'force', 'quiet', 'lenient', 'keep_going', 'no_docstrings']
        for opt_name in empty_containers:
            if len(getattr(options, opt_name)) != 0 and opt_name not in skip:
                self.assertEqual(opt_name, '', msg='For option ' + opt_name)
                return False
        for opt_name in are_none:
            if getattr(options, opt_name) is not None and opt_name not in skip:
                self.assertEqual(opt_name, '', msg='For option ' + opt_name)
                return False
        if options.parallel != parallel_compiles and 'parallel' not in skip:
            return False
        return True

    def test_directive_short(self):
        if False:
            return 10
        (options, args) = self.parse_args(['-X', 'cdivision=True'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['directives']))
        self.assertEqual(options.directives['cdivision'], True)

    def test_directive_long(self):
        if False:
            while True:
                i = 10
        (options, args) = self.parse_args(['--directive', 'cdivision=True'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['directives']))
        self.assertEqual(options.directives['cdivision'], True)

    def test_directive_multiple(self):
        if False:
            return 10
        (options, args) = self.parse_args(['-X', 'cdivision=True', '-X', 'c_string_type=bytes'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['directives']))
        self.assertEqual(options.directives['cdivision'], True)
        self.assertEqual(options.directives['c_string_type'], 'bytes')

    def test_directive_multiple_v2(self):
        if False:
            i = 10
            return i + 15
        (options, args) = self.parse_args(['-X', 'cdivision=True,c_string_type=bytes'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['directives']))
        self.assertEqual(options.directives['cdivision'], True)
        self.assertEqual(options.directives['c_string_type'], 'bytes')

    def test_directive_value_yes(self):
        if False:
            return 10
        (options, args) = self.parse_args(['-X', 'cdivision=YeS'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['directives']))
        self.assertEqual(options.directives['cdivision'], True)

    def test_directive_value_no(self):
        if False:
            for i in range(10):
                print('nop')
        (options, args) = self.parse_args(['-X', 'cdivision=no'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['directives']))
        self.assertEqual(options.directives['cdivision'], False)

    def test_directive_value_invalid(self):
        if False:
            return 10
        with self.assertRaises(ValueError) as context:
            (options, args) = self.parse_args(['-X', 'cdivision=sadfasd'])

    def test_directive_key_invalid(self):
        if False:
            return 10
        with self.assertRaises(ValueError) as context:
            (options, args) = self.parse_args(['-X', 'abracadabra'])

    def test_directive_no_value(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError) as context:
            (options, args) = self.parse_args(['-X', 'cdivision'])

    def test_directives_types(self):
        if False:
            print('Hello World!')
        directives = [('auto_pickle', True), ('c_string_type', 'bytearray'), ('c_string_type', 'bytes'), ('c_string_type', 'str'), ('c_string_type', 'bytearray'), ('c_string_type', 'unicode'), ('c_string_encoding', 'ascii'), ('language_level', '2'), ('language_level', '3'), ('language_level', '3str'), ('set_initial_path', 'my_initial_path')]
        for (key, value) in directives:
            cmd = '{key}={value}'.format(key=key, value=str(value))
            (options, args) = self.parse_args(['-X', cmd])
            self.assertFalse(args)
            self.assertTrue(self.are_default(options, ['directives']), msg='Error for option: ' + cmd)
            self.assertEqual(options.directives[key], value, msg='Error for option: ' + cmd)

    def test_directives_wrong(self):
        if False:
            while True:
                i = 10
        directives = [('auto_pickle', 42), ('auto_pickle', 'NONONO'), ('c_string_type', 'bites')]
        for (key, value) in directives:
            cmd = '{key}={value}'.format(key=key, value=str(value))
            with self.assertRaises(ValueError, msg='Error for option: ' + cmd) as context:
                (options, args) = self.parse_args(['-X', cmd])

    def test_compile_time_env_short(self):
        if False:
            while True:
                i = 10
        (options, args) = self.parse_args(['-E', 'MYSIZE=10'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['compile_time_env']))
        self.assertEqual(options.compile_time_env['MYSIZE'], 10)

    def test_compile_time_env_long(self):
        if False:
            while True:
                i = 10
        (options, args) = self.parse_args(['--compile-time-env', 'MYSIZE=10'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['compile_time_env']))
        self.assertEqual(options.compile_time_env['MYSIZE'], 10)

    def test_compile_time_env_multiple(self):
        if False:
            print('Hello World!')
        (options, args) = self.parse_args(['-E', 'MYSIZE=10', '-E', 'ARRSIZE=11'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['compile_time_env']))
        self.assertEqual(options.compile_time_env['MYSIZE'], 10)
        self.assertEqual(options.compile_time_env['ARRSIZE'], 11)

    def test_compile_time_env_multiple_v2(self):
        if False:
            while True:
                i = 10
        (options, args) = self.parse_args(['-E', 'MYSIZE=10,ARRSIZE=11'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['compile_time_env']))
        self.assertEqual(options.compile_time_env['MYSIZE'], 10)
        self.assertEqual(options.compile_time_env['ARRSIZE'], 11)

    def test_option_short(self):
        if False:
            print('Hello World!')
        (options, args) = self.parse_args(['-s', 'docstrings=True'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['docstrings'], True)

    def test_option_long(self):
        if False:
            for i in range(10):
                print('nop')
        (options, args) = self.parse_args(['--option', 'docstrings=True'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['docstrings'], True)

    def test_option_multiple(self):
        if False:
            i = 10
            return i + 15
        (options, args) = self.parse_args(['-s', 'docstrings=True', '-s', 'buffer_max_dims=8'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['docstrings'], True)
        self.assertEqual(options.options['buffer_max_dims'], True)

    def test_option_multiple_v2(self):
        if False:
            for i in range(10):
                print('nop')
        (options, args) = self.parse_args(['-s', 'docstrings=True,buffer_max_dims=8'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['docstrings'], True)
        self.assertEqual(options.options['buffer_max_dims'], True)

    def test_option_value_yes(self):
        if False:
            return 10
        (options, args) = self.parse_args(['-s', 'docstrings=YeS'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['docstrings'], True)

    def test_option_value_4242(self):
        if False:
            for i in range(10):
                print('nop')
        (options, args) = self.parse_args(['-s', 'docstrings=4242'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['docstrings'], True)

    def test_option_value_0(self):
        if False:
            while True:
                i = 10
        (options, args) = self.parse_args(['-s', 'docstrings=0'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['docstrings'], False)

    def test_option_value_emptystr(self):
        if False:
            i = 10
            return i + 15
        (options, args) = self.parse_args(['-s', 'docstrings='])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['docstrings'], True)

    def test_option_value_a_str(self):
        if False:
            return 10
        (options, args) = self.parse_args(['-s', 'docstrings=BB'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['docstrings'], True)

    def test_option_value_no(self):
        if False:
            for i in range(10):
                print('nop')
        (options, args) = self.parse_args(['-s', 'docstrings=nO'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['docstrings'], False)

    def test_option_no_value(self):
        if False:
            return 10
        (options, args) = self.parse_args(['-s', 'docstrings'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['docstrings'], True)

    def test_option_any_key(self):
        if False:
            i = 10
            return i + 15
        (options, args) = self.parse_args(['-s', 'abracadabra'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['options']))
        self.assertEqual(options.options['abracadabra'], True)

    def test_language_level_2(self):
        if False:
            print('Hello World!')
        (options, args) = self.parse_args(['-2'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['language_level']))
        self.assertEqual(options.language_level, 2)

    def test_language_level_3(self):
        if False:
            print('Hello World!')
        (options, args) = self.parse_args(['-3'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['language_level']))
        self.assertEqual(options.language_level, 3)

    def test_language_level_3str(self):
        if False:
            for i in range(10):
                print('nop')
        (options, args) = self.parse_args(['--3str'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['language_level']))
        self.assertEqual(options.language_level, '3str')

    def test_annotate_short(self):
        if False:
            while True:
                i = 10
        (options, args) = self.parse_args(['-a'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['annotate']))
        self.assertEqual(options.annotate, 'default')

    def test_annotate_long(self):
        if False:
            return 10
        (options, args) = self.parse_args(['--annotate'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['annotate']))
        self.assertEqual(options.annotate, 'default')

    def test_annotate_fullc(self):
        if False:
            print('Hello World!')
        (options, args) = self.parse_args(['--annotate-fullc'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['annotate']))
        self.assertEqual(options.annotate, 'fullc')

    def test_annotate_and_positional(self):
        if False:
            return 10
        (options, args) = self.parse_args(['-a', 'foo.pyx'])
        self.assertEqual(args, ['foo.pyx'])
        self.assertTrue(self.are_default(options, ['annotate']))
        self.assertEqual(options.annotate, 'default')

    def test_annotate_and_optional(self):
        if False:
            return 10
        (options, args) = self.parse_args(['-a', '--3str'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['annotate', 'language_level']))
        self.assertEqual(options.annotate, 'default')
        self.assertEqual(options.language_level, '3str')

    def test_exclude_short(self):
        if False:
            print('Hello World!')
        (options, args) = self.parse_args(['-x', '*.pyx'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['excludes']))
        self.assertTrue('*.pyx' in options.excludes)

    def test_exclude_long(self):
        if False:
            return 10
        (options, args) = self.parse_args(['--exclude', '*.pyx'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['excludes']))
        self.assertTrue('*.pyx' in options.excludes)

    def test_exclude_multiple(self):
        if False:
            return 10
        (options, args) = self.parse_args(['--exclude', '*.pyx', '--exclude', '*.py'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['excludes']))
        self.assertEqual(options.excludes, ['*.pyx', '*.py'])

    def test_build_short(self):
        if False:
            i = 10
            return i + 15
        (options, args) = self.parse_args(['-b'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['build']))
        self.assertEqual(options.build, True)

    def test_build_long(self):
        if False:
            for i in range(10):
                print('nop')
        (options, args) = self.parse_args(['--build'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['build']))
        self.assertEqual(options.build, True)

    def test_inplace_short(self):
        if False:
            while True:
                i = 10
        (options, args) = self.parse_args(['-i'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['build_inplace']))
        self.assertEqual(options.build_inplace, True)

    def test_inplace_long(self):
        if False:
            print('Hello World!')
        (options, args) = self.parse_args(['--inplace'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['build_inplace']))
        self.assertEqual(options.build_inplace, True)

    def test_parallel_short(self):
        if False:
            return 10
        (options, args) = self.parse_args(['-j', '42'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['parallel']))
        self.assertEqual(options.parallel, 42)

    def test_parallel_long(self):
        if False:
            return 10
        (options, args) = self.parse_args(['--parallel', '42'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['parallel']))
        self.assertEqual(options.parallel, 42)

    def test_force_short(self):
        if False:
            for i in range(10):
                print('nop')
        (options, args) = self.parse_args(['-f'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['force']))
        self.assertEqual(options.force, True)

    def test_force_long(self):
        if False:
            while True:
                i = 10
        (options, args) = self.parse_args(['--force'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['force']))
        self.assertEqual(options.force, True)

    def test_quite_short(self):
        if False:
            return 10
        (options, args) = self.parse_args(['-q'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['quiet']))
        self.assertEqual(options.quiet, True)

    def test_quite_long(self):
        if False:
            return 10
        (options, args) = self.parse_args(['--quiet'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['quiet']))
        self.assertEqual(options.quiet, True)

    def test_lenient_long(self):
        if False:
            i = 10
            return i + 15
        (options, args) = self.parse_args(['--lenient'])
        self.assertTrue(self.are_default(options, ['lenient']))
        self.assertFalse(args)
        self.assertEqual(options.lenient, True)

    def test_keep_going_short(self):
        if False:
            for i in range(10):
                print('nop')
        (options, args) = self.parse_args(['-k'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['keep_going']))
        self.assertEqual(options.keep_going, True)

    def test_keep_going_long(self):
        if False:
            print('Hello World!')
        (options, args) = self.parse_args(['--keep-going'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['keep_going']))
        self.assertEqual(options.keep_going, True)

    def test_no_docstrings_long(self):
        if False:
            while True:
                i = 10
        (options, args) = self.parse_args(['--no-docstrings'])
        self.assertFalse(args)
        self.assertTrue(self.are_default(options, ['no_docstrings']))
        self.assertEqual(options.no_docstrings, True)

    def test_file_name(self):
        if False:
            return 10
        (options, args) = self.parse_args(['file1.pyx', 'file2.pyx'])
        self.assertEqual(len(args), 2)
        self.assertEqual(args[0], 'file1.pyx')
        self.assertEqual(args[1], 'file2.pyx')
        self.assertTrue(self.are_default(options, []))

    def test_option_first(self):
        if False:
            print('Hello World!')
        (options, args) = self.parse_args(['-i', 'file.pyx'])
        self.assertEqual(args, ['file.pyx'])
        self.assertEqual(options.build_inplace, True)
        self.assertTrue(self.are_default(options, ['build_inplace']))

    def test_file_inbetween(self):
        if False:
            for i in range(10):
                print('nop')
        (options, args) = self.parse_args(['-i', 'file.pyx', '-a'])
        self.assertEqual(args, ['file.pyx'])
        self.assertEqual(options.build_inplace, True)
        self.assertEqual(options.annotate, 'default')
        self.assertTrue(self.are_default(options, ['build_inplace', 'annotate']))

    def test_option_trailing(self):
        if False:
            print('Hello World!')
        (options, args) = self.parse_args(['file.pyx', '-i'])
        self.assertEqual(args, ['file.pyx'])
        self.assertEqual(options.build_inplace, True)
        self.assertTrue(self.are_default(options, ['build_inplace']))

    def test_interspersed_positional(self):
        if False:
            print('Hello World!')
        (options, sources) = self.parse_args(['file1.pyx', '-a', 'file2.pyx'])
        self.assertEqual(sources, ['file1.pyx', 'file2.pyx'])
        self.assertEqual(options.annotate, 'default')
        self.assertTrue(self.are_default(options, ['annotate']))

    def test_interspersed_positional2(self):
        if False:
            print('Hello World!')
        (options, sources) = self.parse_args(['file1.pyx', '-a', 'file2.pyx', '-a', 'file3.pyx'])
        self.assertEqual(sources, ['file1.pyx', 'file2.pyx', 'file3.pyx'])
        self.assertEqual(options.annotate, 'default')
        self.assertTrue(self.are_default(options, ['annotate']))

    def test_interspersed_positional3(self):
        if False:
            while True:
                i = 10
        (options, sources) = self.parse_args(['-f', 'f1', 'f2', '-a', 'f3', 'f4', '-a', 'f5'])
        self.assertEqual(sources, ['f1', 'f2', 'f3', 'f4', 'f5'])
        self.assertEqual(options.annotate, 'default')
        self.assertEqual(options.force, True)
        self.assertTrue(self.are_default(options, ['annotate', 'force']))

    def test_wrong_option(self):
        if False:
            i = 10
            return i + 15
        old_stderr = sys.stderr
        stderr = sys.stderr = StringIO()
        try:
            self.assertRaises(SystemExit, self.parse_args, ['--unknown-option'])
        finally:
            sys.stderr = old_stderr
        self.assertTrue(stderr.getvalue())

class TestParseArgs(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._options_backup = backup_Options()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        restore_Options(self._options_backup)

    def check_default_global_options(self, white_list=[]):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(check_global_options(self._options_backup, white_list), '')

    def test_build_set_for_inplace(self):
        if False:
            return 10
        (options, args) = parse_args(['foo.pyx', '-i'])
        self.assertEqual(options.build, True)
        self.check_default_global_options()

    def test_lenient(self):
        if False:
            i = 10
            return i + 15
        (options, sources) = parse_args(['foo.pyx', '--lenient'])
        self.assertEqual(sources, ['foo.pyx'])
        self.assertEqual(Options.error_on_unknown_names, False)
        self.assertEqual(Options.error_on_uninitialized, False)
        self.check_default_global_options(['error_on_unknown_names', 'error_on_uninitialized'])

    def test_annotate(self):
        if False:
            i = 10
            return i + 15
        (options, sources) = parse_args(['foo.pyx', '--annotate'])
        self.assertEqual(sources, ['foo.pyx'])
        self.assertEqual(Options.annotate, 'default')
        self.check_default_global_options(['annotate'])

    def test_annotate_fullc(self):
        if False:
            print('Hello World!')
        (options, sources) = parse_args(['foo.pyx', '--annotate-fullc'])
        self.assertEqual(sources, ['foo.pyx'])
        self.assertEqual(Options.annotate, 'fullc')
        self.check_default_global_options(['annotate'])

    def test_no_docstrings(self):
        if False:
            print('Hello World!')
        (options, sources) = parse_args(['foo.pyx', '--no-docstrings'])
        self.assertEqual(sources, ['foo.pyx'])
        self.assertEqual(Options.docstrings, False)
        self.check_default_global_options(['docstrings'])