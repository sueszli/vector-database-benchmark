"""Tests for pytype_runner.py."""
import collections
import dataclasses
import re
import sys
from pytype import config as pytype_config
from pytype import file_utils
from pytype import module_utils
from pytype.platform_utils import path_utils
from pytype.tests import test_utils
from pytype.tools.analyze_project import parse_args
from pytype.tools.analyze_project import pytype_runner
import unittest
Module = module_utils.Module
Action = pytype_runner.Action
Stage = pytype_runner.Stage

@dataclasses.dataclass(eq=True, frozen=True)
class Local:
    path: str
    short_path: str
    module_name: str

@dataclasses.dataclass(eq=True, frozen=True)
class ExpectedBuildStatement:
    output: str
    action: str
    input: str
    deps: str
    imports: str
    module: str
_PREAMBLE_LENGTH = 6

class FakeImportGraph:
    """Just enough of the ImportGraph interface to run tests."""

    def __init__(self, source_files, provenance, source_to_deps):
        if False:
            while True:
                i = 10
        self.source_files = source_files
        self.provenance = provenance
        self.source_to_deps = source_to_deps

    def deps_list(self):
        if False:
            return 10
        return [(x, self.source_to_deps[x]) for x in reversed(self.source_files)]

def make_runner(sources, dep, conf):
    if False:
        i = 10
        return i + 15
    conf.inputs = [m.full_path for m in sources]
    return pytype_runner.PytypeRunner(conf, dep)

class TestResolvedFileToModule(unittest.TestCase):
    """Test resolved_file_to_module."""

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        resolved_file = Local('foo/bar.py', 'bar.py', 'bar')
        self.assertEqual(pytype_runner.resolved_file_to_module(resolved_file), Module('foo/', 'bar.py', 'bar', 'Local'))

    def test_preserve_init(self):
        if False:
            while True:
                i = 10
        resolved_file = Local('foo/bar/__init__.py', 'bar/__init__.py', 'bar')
        self.assertEqual(pytype_runner.resolved_file_to_module(resolved_file), Module('foo/', 'bar/__init__.py', 'bar.__init__', 'Local'))

class TestDepsFromImportGraph(unittest.TestCase):
    """Test deps_from_import_graph."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        init = Local('/foo/bar/__init__.py', 'bar/__init__.py', 'bar')
        a = Local('/foo/bar/a.py', 'bar/a.py', 'bar.a')
        b = Local('/foo/bar/b.py', 'bar/b.py', 'bar.b')
        self.sources = [x.path for x in [init, a, b]]
        self.provenance = {x.path: x for x in [init, a, b]}

    def test_basic(self):
        if False:
            while True:
                i = 10
        graph = FakeImportGraph(self.sources, self.provenance, collections.defaultdict(list))
        deps = pytype_runner.deps_from_import_graph(graph)
        expected = [((Module('/foo/', 'bar/__init__.py', 'bar.__init__'),), ()), ((Module('/foo/', 'bar/a.py', 'bar.a'),), ()), ((Module('/foo/', 'bar/b.py', 'bar.b'),), ())]
        self.assertEqual(deps, expected)

    def test_duplicate_deps(self):
        if False:
            print('Hello World!')
        graph = FakeImportGraph(self.sources, self.provenance, collections.defaultdict(lambda : [self.sources[0]] * 2))
        deps = pytype_runner.deps_from_import_graph(graph)
        init = Module('/foo/', 'bar/__init__.py', 'bar.__init__')
        expected = [((init,), (init,)), ((Module('/foo/', 'bar/a.py', 'bar.a'),), (init,)), ((Module('/foo/', 'bar/b.py', 'bar.b'),), (init,))]
        self.assertEqual(deps, expected)

    def test_pyi_src(self):
        if False:
            for i in range(10):
                print('nop')
        pyi_mod = Local('/foo/bar/c.pyi', 'bar/c.pyi', 'bar.c')
        provenance = {pyi_mod.path: pyi_mod}
        provenance.update(self.provenance)
        graph = FakeImportGraph(self.sources + [pyi_mod.path], provenance, collections.defaultdict(list))
        deps = pytype_runner.deps_from_import_graph(graph)
        expected = [((Module('/foo/', 'bar/__init__.py', 'bar.__init__'),), ()), ((Module('/foo/', 'bar/a.py', 'bar.a'),), ()), ((Module('/foo/', 'bar/b.py', 'bar.b'),), ())]
        self.assertEqual(deps, expected)

    def test_pyi_dep(self):
        if False:
            return 10
        pyi_mod = Local('/foo/bar/c.pyi', 'bar/c.pyi', 'bar.c')
        graph = FakeImportGraph(self.sources, self.provenance, collections.defaultdict(lambda : [pyi_mod.path]))
        deps = pytype_runner.deps_from_import_graph(graph)
        expected = [((Module('/foo/', 'bar/__init__.py', 'bar.__init__'),), ()), ((Module('/foo/', 'bar/a.py', 'bar.a'),), ()), ((Module('/foo/', 'bar/b.py', 'bar.b'),), ())]
        self.assertEqual(deps, expected)

    def test_pyi_with_src_dep(self):
        if False:
            i = 10
            return i + 15
        py_mod = Local('/foo/a/b.py', 'a/b.py', 'a.b')
        pyi_mod = Local('/foo/bar/c.pyi', 'bar/c.pyi', 'bar.c')
        py_dep = Local('/foo/a/c.py', 'a/c.py', 'a.c')
        sources = [py_dep, pyi_mod, py_mod]
        graph = FakeImportGraph(source_files=[x.path for x in sources], provenance={x.path: x for x in sources}, source_to_deps={py_mod.path: [pyi_mod.path], pyi_mod.path: [py_dep.path], py_dep.path: []})
        deps = pytype_runner.deps_from_import_graph(graph)
        expected = [((Module('/foo/', 'a/c.py', 'a.c'),), ()), ((Module('/foo/', 'a/b.py', 'a.b'),), (Module('/foo/', 'a/c.py', 'a.c'),))]
        self.assertEqual(deps, expected)

    def test_pyi_with_src_dep_transitive(self):
        if False:
            i = 10
            return i + 15
        py_mod = Local('/foo/a/b.py', 'a/b.py', 'a.b')
        pyi_mod = Local('/foo/bar/c.pyi', 'bar/c.pyi', 'bar.c')
        pyi_dep = Local('/foo/bar/d.pyi', 'bar/d.pyi', 'bar.d')
        py_dep = Local('/foo/a/c.py', 'a/c.py', 'a.c')
        sources = [py_dep, pyi_dep, pyi_mod, py_mod]
        graph = FakeImportGraph(source_files=[x.path for x in sources], provenance={x.path: x for x in sources}, source_to_deps={py_mod.path: [pyi_mod.path], pyi_mod.path: [pyi_dep.path], pyi_dep.path: [py_dep.path], py_dep.path: []})
        deps = pytype_runner.deps_from_import_graph(graph)
        expected = [((Module('/foo/', 'a/c.py', 'a.c'),), ()), ((Module('/foo/', 'a/b.py', 'a.b'),), (Module('/foo/', 'a/c.py', 'a.c'),))]
        self.assertEqual(deps, expected)

    def test_pyi_with_src_dep_branching(self):
        if False:
            print('Hello World!')
        py_mod = Local('/foo/a/b.py', 'a/b.py', 'a.b')
        pyi_mod1 = Local('/foo/bar/c.pyi', 'bar/c.pyi', 'bar.c')
        py_dep1 = Local('/foo/a/c.py', 'a/c.py', 'a.c')
        py_dep2 = Local('/foo/a/d.py', 'a/d.py', 'a.d')
        pyi_mod2 = Local('/foo/bar/d.pyi', 'bar/d.pyi', 'bar.d')
        py_dep3 = Local('/foo/a/e.py', 'a/e.py', 'a.e')
        sources = [py_dep3, pyi_mod2, py_dep2, py_dep1, pyi_mod1, py_mod]
        graph = FakeImportGraph(source_files=[x.path for x in sources], provenance={x.path: x for x in sources}, source_to_deps={py_mod.path: [pyi_mod1.path, pyi_mod2.path], pyi_mod1.path: [py_dep1.path, py_dep2.path], py_dep1.path: [], py_dep2.path: [], pyi_mod2.path: [py_dep3.path], py_dep3.path: []})
        deps = pytype_runner.deps_from_import_graph(graph)
        expected = [((Module('/foo/', 'a/e.py', 'a.e'),), ()), ((Module('/foo/', 'a/d.py', 'a.d'),), ()), ((Module('/foo/', 'a/c.py', 'a.c'),), ()), ((Module('/foo/', 'a/b.py', 'a.b'),), (Module('/foo/', 'a/c.py', 'a.c'), Module('/foo/', 'a/d.py', 'a.d'), Module('/foo/', 'a/e.py', 'a.e')))]
        self.assertEqual(deps, expected)

class TestBase(unittest.TestCase):
    """Base class for tests using a parser."""

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        cls.parser = parse_args.make_parser()

class TestCustomOptions(TestBase):
    """Test PytypeRunner.set_custom_options."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.conf = self.parser.config_from_defaults()

    def assertFlags(self, flags, expected_flags):
        if False:
            while True:
                i = 10
        temporary_flags = set()
        self.assertEqual(flags - temporary_flags, expected_flags)

    def test_disable(self):
        if False:
            while True:
                i = 10
        self.conf.disable = ['import-error', 'name-error']
        runner = make_runner([], [], self.conf)
        flags_with_values = {}
        runner.set_custom_options(flags_with_values, set(), self.conf.report_errors)
        self.assertEqual(flags_with_values['--disable'], 'import-error,name-error')

    def test_no_disable(self):
        if False:
            return 10
        self.conf.disable = []
        runner = make_runner([], [], self.conf)
        flags_with_values = {}
        runner.set_custom_options(flags_with_values, set(), self.conf.report_errors)
        self.assertFalse(flags_with_values)

    def test_report_errors(self):
        if False:
            while True:
                i = 10
        self.conf.report_errors = True
        runner = make_runner([], [], self.conf)
        binary_flags = {'--no-report-errors'}
        runner.set_custom_options({}, binary_flags, True)
        self.assertFlags(binary_flags, set())

    def test_no_report_errors(self):
        if False:
            i = 10
            return i + 15
        self.conf.report_errors = False
        runner = make_runner([], [], self.conf)
        binary_flags = set()
        runner.set_custom_options({}, binary_flags, True)
        self.assertFlags(binary_flags, {'--no-report-errors'})

    def test_report_errors_default(self):
        if False:
            for i in range(10):
                print('nop')
        self.conf.report_errors = True
        runner = make_runner([], [], self.conf)
        binary_flags = set()
        runner.set_custom_options({}, binary_flags, True)
        self.assertFlags(binary_flags, set())

    def test_protocols(self):
        if False:
            return 10
        self.conf.protocols = True
        runner = make_runner([], [], self.conf)
        binary_flags = set()
        runner.set_custom_options({}, binary_flags, self.conf.report_errors)
        self.assertFlags(binary_flags, {'--protocols'})

    def test_no_protocols(self):
        if False:
            print('Hello World!')
        self.conf.protocols = False
        runner = make_runner([], [], self.conf)
        binary_flags = {'--protocols'}
        runner.set_custom_options({}, binary_flags, self.conf.report_errors)
        self.assertFlags(binary_flags, set())

    def test_no_protocols_default(self):
        if False:
            print('Hello World!')
        self.conf.protocols = False
        runner = make_runner([], [], self.conf)
        binary_flags = set()
        runner.set_custom_options({}, binary_flags, self.conf.report_errors)
        self.assertFlags(binary_flags, set())

class TestGetRunCmd(TestBase):
    """Test PytypeRunner.get_pytype_command_for_ninja()."""

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.runner = make_runner([], [], self.parser.config_from_defaults())

    def get_options(self, args):
        if False:
            return 10
        nargs = len(pytype_runner.PYTYPE_SINGLE)
        self.assertEqual(args[:nargs], pytype_runner.PYTYPE_SINGLE)
        args = args[nargs:]
        (start, end) = (args.index('--imports_info'), args.index('$imports'))
        self.assertEqual(end - start, 1)
        args.pop(end)
        args.pop(start)
        return pytype_config.Options(args, command_line=True)

    def get_basic_options(self, report_errors=False):
        if False:
            return 10
        return self.get_options(self.runner.get_pytype_command_for_ninja(report_errors))

    def test_python_version(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.get_basic_options().python_version, tuple((int(i) for i in self.runner.python_version.split('.'))))

    def test_output(self):
        if False:
            return 10
        self.assertEqual(self.get_basic_options().output, '$out')

    def test_quick(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.get_basic_options().quick)

    def test_module_name(self):
        if False:
            return 10
        self.assertEqual(self.get_basic_options().module_name, '$module')

    def test_error_reporting(self):
        if False:
            while True:
                i = 10
        options = self.get_basic_options(report_errors=False)
        self.assertFalse(options.report_errors)
        self.assertFalse(options.analyze_annotated)
        options = self.get_basic_options(report_errors=True)
        self.assertTrue(options.report_errors)
        self.assertTrue(options.analyze_annotated)

    def test_custom_option(self):
        if False:
            for i in range(10):
                print('nop')
        custom_conf = self.parser.config_from_defaults()
        custom_conf.disable = ['import-error', 'name-error']
        self.runner = make_runner([], [], custom_conf)
        args = self.runner.get_pytype_command_for_ninja(report_errors=True)
        options = self.get_options(args)
        self.assertEqual(options.disable, ['import-error', 'name-error'])

    def test_custom_option_no_report_errors(self):
        if False:
            while True:
                i = 10
        custom_conf = self.parser.config_from_defaults()
        custom_conf.precise_return = True
        self.runner = make_runner([], [], custom_conf)
        args = self.runner.get_pytype_command_for_ninja(report_errors=False)
        options = self.get_options(args)
        self.assertTrue(options.precise_return)

class TestGetModuleAction(TestBase):
    """Tests for PytypeRunner.get_module_action."""

    def test_check(self):
        if False:
            return 10
        sources = [Module('', 'foo.py', 'foo')]
        runner = make_runner(sources, [], self.parser.config_from_defaults())
        self.assertEqual(runner.get_module_action(sources[0]), pytype_runner.Action.CHECK)

    def test_infer(self):
        if False:
            while True:
                i = 10
        runner = make_runner([], [], self.parser.config_from_defaults())
        self.assertEqual(runner.get_module_action(Module('', 'foo.py', 'foo')), pytype_runner.Action.INFER)

    def test_generate_default(self):
        if False:
            print('Hello World!')
        runner = make_runner([], [], self.parser.config_from_defaults())
        self.assertEqual(runner.get_module_action(Module('', 'foo.py', 'foo', 'System')), pytype_runner.Action.GENERATE_DEFAULT)

class TestYieldSortedModules(TestBase):
    """Tests for PytypeRunner.yield_sorted_modules()."""

    def normalize(self, d):
        if False:
            for i in range(10):
                print('nop')
        return file_utils.expand_path(d).rstrip(path_utils.sep) + path_utils.sep

    def assert_sorted_modules_equal(self, mod_gen, expected_list):
        if False:
            print('Hello World!')
        for (expected_mod, expected_report_errors, expected_deps, expected_stage) in expected_list:
            try:
                (mod, actual_report_errors, actual_deps, actual_stage) = next(mod_gen)
            except StopIteration as e:
                raise AssertionError('Not enough modules') from e
            self.assertEqual(mod, expected_mod)
            self.assertEqual(actual_report_errors, expected_report_errors)
            self.assertEqual(actual_deps, expected_deps)
            self.assertEqual(actual_stage, expected_stage)
        try:
            next(mod_gen)
        except StopIteration:
            pass
        else:
            raise AssertionError('Too many modules')

    def test_source(self):
        if False:
            while True:
                i = 10
        conf = self.parser.config_from_defaults()
        d = self.normalize('foo/')
        conf.pythonpath = [d]
        f = Module(d, 'bar.py', 'bar')
        runner = make_runner([f], [((f,), ())], conf)
        self.assert_sorted_modules_equal(runner.yield_sorted_modules(), [(f, Action.CHECK, (), Stage.SINGLE_PASS)])

    def test_source_and_dep(self):
        if False:
            print('Hello World!')
        conf = self.parser.config_from_defaults()
        d = self.normalize('foo/')
        conf.pythonpath = [d]
        src = Module(d, 'bar.py', 'bar')
        dep = Module(d, 'baz.py', 'baz')
        runner = make_runner([src], [((dep,), ()), ((src,), (dep,))], conf)
        self.assert_sorted_modules_equal(runner.yield_sorted_modules(), [(dep, Action.INFER, (), Stage.SINGLE_PASS), (src, Action.CHECK, (dep,), Stage.SINGLE_PASS)])

    def test_cycle(self):
        if False:
            for i in range(10):
                print('nop')
        conf = self.parser.config_from_defaults()
        d = self.normalize('foo/')
        conf.pythonpath = [d]
        src = Module(d, 'bar.py', 'bar')
        dep = Module(d, 'baz.py', 'baz')
        runner = make_runner([src], [((dep, src), ())], conf)
        self.assert_sorted_modules_equal(runner.yield_sorted_modules(), [(dep, Action.INFER, (), Stage.FIRST_PASS), (src, Action.INFER, (), Stage.FIRST_PASS), (dep, Action.INFER, (dep, src), Stage.SECOND_PASS), (src, Action.CHECK, (dep, src), Stage.SECOND_PASS)])

    def test_system_dep(self):
        if False:
            return 10
        conf = self.parser.config_from_defaults()
        d = self.normalize('foo/')
        external = self.normalize('quux/')
        conf.pythonpath = [d]
        mod = Module(external, 'bar/baz.py', 'bar.baz', 'System')
        runner = make_runner([], [((mod,), ())], conf)
        self.assert_sorted_modules_equal(runner.yield_sorted_modules(), [(mod, Action.GENERATE_DEFAULT, (), Stage.SINGLE_PASS)])

class TestNinjaPathEscape(TestBase):

    def test_escape(self):
        if False:
            print('Hello World!')
        escaped = pytype_runner.escape_ninja_path('C:/xyz')
        if sys.platform == 'win32':
            self.assertEqual(escaped, 'C$:/xyz')
        else:
            self.assertEqual(escaped, 'C:/xyz')

    def test_already_escaped(self):
        if False:
            while True:
                i = 10
        self.assertEqual(pytype_runner.escape_ninja_path('C$:/xyz'), 'C$:/xyz')

class TestNinjaPreamble(TestBase):
    """Tests for PytypeRunner.write_ninja_preamble."""

    def test_write(self):
        if False:
            i = 10
            return i + 15
        conf = self.parser.config_from_defaults()
        with test_utils.Tempdir() as d:
            conf.output = d.path
            runner = make_runner([], [], conf)
            runner.write_ninja_preamble()
            with open(runner.ninja_file) as f:
                preamble = f.read().splitlines()
        self.assertEqual(len(preamble), _PREAMBLE_LENGTH)
        for (i, line) in enumerate(preamble):
            if not i % 3:
                self.assertRegex(line, 'rule \\w*')
            elif i % 3 == 1:
                expected = '  command = {} .* \\$in'.format(re.escape(' '.join(pytype_runner.PYTYPE_SINGLE)))
                self.assertRegex(line, expected)
            else:
                self.assertRegex(line, '  description = \\w* \\$module')

class TestNinjaBuildStatement(TestBase):
    """Tests for PytypeRunner.write_build_statement."""

    def write_build_statement(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        conf = self.parser.config_from_defaults()
        with test_utils.Tempdir() as d:
            conf.output = d.path
            runner = make_runner([], [], conf)
            output = runner.write_build_statement(*args, **kwargs)
            with open(runner.ninja_file) as f:
                return (runner, output, f.read().splitlines())

    def assertOutputMatches(self, module, expected_output):
        if False:
            return 10
        (runner, output, _) = self.write_build_statement(module, Action.CHECK, set(), 'imports', '')
        self.assertEqual(output, path_utils.join(runner.pyi_dir, expected_output))

    def test_check(self):
        if False:
            i = 10
            return i + 15
        (_, output, build_statement) = self.write_build_statement(Module('', 'foo.py', 'foo'), Action.CHECK, set(), 'imports', '')
        self.assertEqual(build_statement[0], f'build {pytype_runner.escape_ninja_path(output)}: check foo.py')

    def test_infer(self):
        if False:
            print('Hello World!')
        (_, output, build_statement) = self.write_build_statement(Module('', 'foo.py', 'foo'), Action.INFER, set(), 'imports', '')
        self.assertEqual(build_statement[0], f'build {pytype_runner.escape_ninja_path(output)}: infer foo.py')

    def test_deps(self):
        if False:
            i = 10
            return i + 15
        (_, output, _) = self.write_build_statement(Module('', 'foo.py', 'foo'), Action.INFER, set(), 'imports', '')
        (_, _, build_statement) = self.write_build_statement(Module('', 'bar.py', 'bar'), Action.CHECK, {pytype_runner.escape_ninja_path(output)}, 'imports', '')
        self.assertTrue(build_statement[0].endswith(' | ' + pytype_runner.escape_ninja_path(output)))

    def test_imports(self):
        if False:
            return 10
        (_, _, build_statement) = self.write_build_statement(Module('', 'foo.py', 'foo'), Action.CHECK, set(), 'imports', '')
        self.assertIn('  imports = imports', build_statement)

    def test_module(self):
        if False:
            while True:
                i = 10
        (_, _, build_statement) = self.write_build_statement(Module('', 'foo.py', 'foo'), Action.CHECK, set(), 'imports', '')
        self.assertIn('  module = foo', build_statement)

    def test_suffix(self):
        if False:
            return 10
        (runner, output, _) = self.write_build_statement(Module('', 'foo.py', 'foo'), Action.CHECK, set(), 'imports', '-1')
        self.assertEqual(path_utils.join(runner.pyi_dir, 'foo.pyi-1'), output)

    def test_hidden_dir(self):
        if False:
            print('Hello World!')
        self.assertOutputMatches(Module('', file_utils.replace_separator('.foo/bar.py'), '.foo.bar'), path_utils.join('.foo', 'bar.pyi'))

    def test_hidden_file(self):
        if False:
            while True:
                i = 10
        self.assertOutputMatches(Module('', file_utils.replace_separator('foo/.bar.py'), 'foo..bar'), path_utils.join('foo', '.bar.pyi'))

    def test_hidden_file_with_path_prefix(self):
        if False:
            i = 10
            return i + 15
        self.assertOutputMatches(Module('', file_utils.replace_separator('foo/.bar.py'), '.bar'), path_utils.join('.bar.pyi'))

    def test_hidden_dir_with_path_mismatch(self):
        if False:
            while True:
                i = 10
        self.assertOutputMatches(Module('', file_utils.replace_separator('symlinked/foo.py'), '.bar'), '.bar.pyi')

    def test_path_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.assertOutputMatches(Module('', file_utils.replace_separator('symlinked/foo.py'), 'bar.baz'), path_utils.join('bar', 'baz.pyi'))

class TestNinjaBody(TestBase):
    """Test PytypeRunner.setup_build."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.conf = self.parser.config_from_defaults()

    def assertBuildStatementMatches(self, build_statement, expected):
        if False:
            i = 10
            return i + 15
        self.assertEqual(build_statement[0], 'build {output}: {action} {input}{deps}'.format(output=pytype_runner.escape_ninja_path(expected.output), action=expected.action, input=pytype_runner.escape_ninja_path(expected.input), deps=pytype_runner.escape_ninja_path(expected.deps)))
        self.assertEqual(set(build_statement[1:]), {f'  imports = {pytype_runner.escape_ninja_path(expected.imports)}', f'  module = {pytype_runner.escape_ninja_path(expected.module)}'})

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        src = Module('', 'foo.py', 'foo')
        dep = Module('', 'bar.py', 'bar')
        with test_utils.Tempdir() as d:
            self.conf.output = d.path
            runner = make_runner([src], [((dep,), ()), ((src,), (dep,))], self.conf)
            runner.setup_build()
            with open(runner.ninja_file) as f:
                body = f.read().splitlines()[_PREAMBLE_LENGTH:]
        self.assertBuildStatementMatches(body[0:3], ExpectedBuildStatement(output=path_utils.join(runner.pyi_dir, 'bar.pyi'), action=Action.INFER, input='bar.py', deps='', imports=path_utils.join(runner.imports_dir, 'bar.imports'), module='bar'))
        self.assertBuildStatementMatches(body[3:], ExpectedBuildStatement(output=path_utils.join(runner.pyi_dir, 'foo.pyi'), action=Action.CHECK, input='foo.py', deps=' | ' + path_utils.join(runner.pyi_dir, 'bar.pyi'), imports=path_utils.join(runner.imports_dir, 'foo.imports'), module='foo'))

    def test_generate_default(self):
        if False:
            for i in range(10):
                print('nop')
        src = Module('', 'foo.py', 'foo')
        dep = Module('', 'bar.py', 'bar', 'System')
        with test_utils.Tempdir() as d:
            self.conf.output = d.path
            runner = make_runner([src], [((dep,), ()), ((src,), (dep,))], self.conf)
            runner.setup_build()
            with open(runner.ninja_file) as f:
                body = f.read().splitlines()[_PREAMBLE_LENGTH:]
            with open(path_utils.join(runner.imports_dir, 'foo.imports')) as f:
                (imports_info,) = f.read().splitlines()
        self.assertBuildStatementMatches(body, ExpectedBuildStatement(output=path_utils.join(runner.pyi_dir, 'foo.pyi'), action=Action.CHECK, input='foo.py', deps='', imports=path_utils.join(runner.imports_dir, 'foo.imports'), module='foo'))
        (short_bar_path, bar_path) = imports_info.split(' ')
        self.assertEqual(short_bar_path, 'bar')
        self.assertEqual(bar_path, path_utils.join(runner.imports_dir, 'default.pyi'))

    def test_cycle(self):
        if False:
            while True:
                i = 10
        src = Module('', 'foo.py', 'foo')
        dep = Module('', 'bar.py', 'bar')
        with test_utils.Tempdir() as d:
            self.conf.output = d.path
            runner = make_runner([src], [((dep, src), ())], self.conf)
            runner.setup_build()
            with open(runner.ninja_file) as f:
                body = f.read().splitlines()[_PREAMBLE_LENGTH:]
        self.assertBuildStatementMatches(body[:3], ExpectedBuildStatement(output=path_utils.join(runner.pyi_dir, 'bar.pyi-1'), action=Action.INFER, input='bar.py', deps='', imports=path_utils.join(runner.imports_dir, 'bar.imports-1'), module='bar'))
        self.assertBuildStatementMatches(body[3:6], ExpectedBuildStatement(output=path_utils.join(runner.pyi_dir, 'foo.pyi-1'), action=Action.INFER, input='foo.py', deps='', imports=path_utils.join(runner.imports_dir, 'foo.imports-1'), module='foo'))
        self.assertBuildStatementMatches(body[6:9], ExpectedBuildStatement(output=path_utils.join(runner.pyi_dir, 'bar.pyi'), action=Action.INFER, input='bar.py', deps=' | {} {}'.format(path_utils.join(runner.pyi_dir, 'bar.pyi-1'), path_utils.join(runner.pyi_dir, 'foo.pyi-1')), imports=path_utils.join(runner.imports_dir, 'bar.imports'), module='bar'))
        self.assertBuildStatementMatches(body[9:], ExpectedBuildStatement(output=path_utils.join(runner.pyi_dir, 'foo.pyi'), action=Action.CHECK, input='foo.py', deps=' | {} {}'.format(path_utils.join(runner.pyi_dir, 'bar.pyi'), path_utils.join(runner.pyi_dir, 'foo.pyi-1')), imports=path_utils.join(runner.imports_dir, 'foo.imports'), module='foo'))

    def test_cycle_with_extra_action(self):
        if False:
            print('Hello World!')
        src = Module('', 'foo.py', 'foo')
        dep = Module('', 'bar.py', 'bar')
        with test_utils.Tempdir() as d:
            self.conf.output = d.path
            runner = make_runner([src], [((src, dep), ())], self.conf)
            runner.setup_build()
            with open(runner.ninja_file) as f:
                body = f.read().splitlines()[_PREAMBLE_LENGTH:]
        self.assertBuildStatementMatches(body[:3], ExpectedBuildStatement(output=path_utils.join(runner.pyi_dir, 'foo.pyi-1'), action=Action.INFER, input='foo.py', deps='', imports=path_utils.join(runner.imports_dir, 'foo.imports-1'), module='foo'))
        self.assertBuildStatementMatches(body[3:6], ExpectedBuildStatement(output=path_utils.join(runner.pyi_dir, 'bar.pyi-1'), action=Action.INFER, input='bar.py', deps='', imports=path_utils.join(runner.imports_dir, 'bar.imports-1'), module='bar'))
        self.assertBuildStatementMatches(body[6:], ExpectedBuildStatement(output=path_utils.join(runner.pyi_dir, 'foo.pyi'), action=Action.CHECK, input='foo.py', deps=' | {} {}'.format(path_utils.join(runner.pyi_dir, 'foo.pyi-1'), path_utils.join(runner.pyi_dir, 'bar.pyi-1')), imports=path_utils.join(runner.imports_dir, 'foo.imports'), module='foo'))

class TestImports(TestBase):
    """Test imports-related functionality."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.conf = self.parser.config_from_defaults()

    def test_write_default_pyi(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            self.conf.output = d.path
            runner = make_runner([], [], self.conf)
            self.assertTrue(runner.make_imports_dir())
            output = runner.write_default_pyi()
            self.assertEqual(output, path_utils.join(runner.imports_dir, 'default.pyi'))
            with open(output) as f:
                self.assertEqual(f.read(), pytype_runner.DEFAULT_PYI)

    def test_write_imports(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            self.conf.output = d.path
            runner = make_runner([], [], self.conf)
            self.assertTrue(runner.make_imports_dir())
            output = runner.write_imports('mod', {'a': 'b'}, '')
            self.assertEqual(path_utils.join(runner.imports_dir, 'mod.imports'), output)
            with open(output) as f:
                self.assertEqual(f.read(), 'a b\n')

    def test_get_imports_map(self):
        if False:
            print('Hello World!')
        mod = Module('', 'foo.py', 'foo')
        deps = (mod,)
        module_to_imports_map = {mod: {'bar': '/dir/bar.pyi'}}
        module_to_output = {mod: '/dir/foo.pyi'}
        imports_map = pytype_runner.get_imports_map(deps, module_to_imports_map, module_to_output)
        self.assertEqual(imports_map, {'foo': '/dir/foo.pyi', 'bar': '/dir/bar.pyi'})
if __name__ == '__main__':
    unittest.main()