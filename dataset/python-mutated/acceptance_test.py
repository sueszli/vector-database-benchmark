import dataclasses
import importlib.metadata
import os
import sys
import types
import pytest
from _pytest.config import ExitCode
from _pytest.pathlib import symlink_or_skip
from _pytest.pytester import Pytester

def prepend_pythonpath(*dirs) -> str:
    if False:
        return 10
    cur = os.getenv('PYTHONPATH')
    if cur:
        dirs += (cur,)
    return os.pathsep.join((str(p) for p in dirs))

class TestGeneralUsage:

    def test_config_error(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.copy_example('conftest_usageerror/conftest.py')
        result = pytester.runpytest(pytester.path)
        assert result.ret == ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(['*ERROR: hello'])
        result.stdout.fnmatch_lines(['*pytest_unconfigure_called'])

    def test_root_conftest_syntax_error(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile(conftest='raise SyntaxError\n')
        result = pytester.runpytest()
        result.stderr.fnmatch_lines(['*raise SyntaxError*'])
        assert result.ret != 0

    def test_early_hook_error_issue38_1(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makeconftest('\n            def pytest_sessionstart():\n                0 / 0\n        ')
        result = pytester.runpytest(pytester.path)
        assert result.ret != 0
        result.stdout.fnmatch_lines(['*INTERNALERROR*File*conftest.py*line 2*', '*0 / 0*'])
        result = pytester.runpytest(pytester.path, '--fulltrace')
        assert result.ret != 0
        result.stdout.fnmatch_lines(['*INTERNALERROR*def pytest_sessionstart():*', '*INTERNALERROR*0 / 0*'])

    def test_early_hook_configure_error_issue38(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makeconftest('\n            def pytest_configure():\n                0 / 0\n        ')
        result = pytester.runpytest(pytester.path)
        assert result.ret != 0
        result.stderr.fnmatch_lines(['*INTERNALERROR*File*conftest.py*line 2*', '*0 / 0*'])

    def test_file_not_found(self, pytester: Pytester) -> None:
        if False:
            return 10
        result = pytester.runpytest('asd')
        assert result.ret != 0
        result.stderr.fnmatch_lines(['ERROR: file or directory not found: asd'])

    def test_file_not_found_unconfigure_issue143(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makeconftest('\n            def pytest_configure():\n                print("---configure")\n            def pytest_unconfigure():\n                print("---unconfigure")\n        ')
        result = pytester.runpytest('-s', 'asd')
        assert result.ret == ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(['ERROR: file or directory not found: asd'])
        result.stdout.fnmatch_lines(['*---configure', '*---unconfigure'])

    def test_config_preparse_plugin_option(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile(pytest_xyz='\n            def pytest_addoption(parser):\n                parser.addoption("--xyz", dest="xyz", action="store")\n        ')
        pytester.makepyfile(test_one='\n            def test_option(pytestconfig):\n                assert pytestconfig.option.xyz == "123"\n        ')
        result = pytester.runpytest('-p', 'pytest_xyz', '--xyz=123', syspathinsert=True)
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*1 passed*'])

    @pytest.mark.parametrize('load_cov_early', [True, False])
    def test_early_load_setuptools_name(self, pytester: Pytester, monkeypatch, load_cov_early) -> None:
        if False:
            while True:
                i = 10
        monkeypatch.delenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD')
        pytester.makepyfile(mytestplugin1_module='')
        pytester.makepyfile(mytestplugin2_module='')
        pytester.makepyfile(mycov_module='')
        pytester.syspathinsert()
        loaded = []

        @dataclasses.dataclass
        class DummyEntryPoint:
            name: str
            module: str
            group: str = 'pytest11'

            def load(self):
                if False:
                    return 10
                __import__(self.module)
                loaded.append(self.name)
                return sys.modules[self.module]
        entry_points = [DummyEntryPoint('myplugin1', 'mytestplugin1_module'), DummyEntryPoint('myplugin2', 'mytestplugin2_module'), DummyEntryPoint('mycov', 'mycov_module')]

        @dataclasses.dataclass
        class DummyDist:
            entry_points: object
            files: object = ()

        def my_dists():
            if False:
                print('Hello World!')
            return (DummyDist(entry_points),)
        monkeypatch.setattr(importlib.metadata, 'distributions', my_dists)
        params = ('-p', 'mycov') if load_cov_early else ()
        pytester.runpytest_inprocess(*params)
        if load_cov_early:
            assert loaded == ['mycov', 'myplugin1', 'myplugin2']
        else:
            assert loaded == ['myplugin1', 'myplugin2', 'mycov']

    @pytest.mark.parametrize('import_mode', ['prepend', 'append', 'importlib'])
    def test_assertion_rewrite(self, pytester: Pytester, import_mode) -> None:
        if False:
            while True:
                i = 10
        p = pytester.makepyfile('\n            def test_this():\n                x = 0\n                assert x\n        ')
        result = pytester.runpytest(p, f'--import-mode={import_mode}')
        result.stdout.fnmatch_lines(['>       assert x', 'E       assert 0'])
        assert result.ret == 1

    def test_nested_import_error(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = pytester.makepyfile('\n                import import_fails\n                def test_this():\n                    assert import_fails.a == 1\n        ')
        pytester.makepyfile(import_fails='import does_not_work')
        result = pytester.runpytest(p)
        result.stdout.fnmatch_lines(['ImportError while importing test module*', '*No module named *does_not_work*'])
        assert result.ret == 2

    def test_not_collectable_arguments(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p1 = pytester.makepyfile('')
        p2 = pytester.makefile('.pyc', '123')
        result = pytester.runpytest(p1, p2)
        assert result.ret == ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines([f'ERROR: found no collectors for {p2}', ''])

    @pytest.mark.filterwarnings('default')
    def test_better_reporting_on_conftest_load_failure(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        'Show a user-friendly traceback on conftest import failures (#486, #3332)'
        pytester.makepyfile('')
        conftest = pytester.makeconftest('\n            def foo():\n                import qwerty\n            foo()\n        ')
        result = pytester.runpytest('--help')
        result.stdout.fnmatch_lines('\n            *--version*\n            *warning*conftest.py*\n        ')
        result = pytester.runpytest()
        assert result.stdout.lines == []
        assert result.stderr.lines == [f"ImportError while loading conftest '{conftest}'.", 'conftest.py:3: in <module>', '    foo()', 'conftest.py:2: in foo', '    import qwerty', "E   ModuleNotFoundError: No module named 'qwerty'"]

    def test_early_skip(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.mkdir('xyz')
        pytester.makeconftest('\n            import pytest\n            def pytest_collect_file():\n                pytest.skip("early")\n        ')
        result = pytester.runpytest()
        assert result.ret == ExitCode.NO_TESTS_COLLECTED
        result.stdout.fnmatch_lines(['*1 skip*'])

    def test_issue88_initial_file_multinodes(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.copy_example('issue88_initial_file_multinodes')
        p = pytester.makepyfile('def test_hello(): pass')
        result = pytester.runpytest(p, '--collect-only')
        result.stdout.fnmatch_lines(['*MyFile*test_issue88*', '*Module*test_issue88*'])

    def test_issue93_initialnode_importing_capturing(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makeconftest('\n            import sys\n            print("should not be seen")\n            sys.stderr.write("stder42\\n")\n        ')
        result = pytester.runpytest()
        assert result.ret == ExitCode.NO_TESTS_COLLECTED
        result.stdout.no_fnmatch_line('*should not be seen*')
        assert 'stderr42' not in result.stderr.str()

    def test_conftest_printing_shows_if_error(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makeconftest('\n            print("should be seen")\n            assert 0\n        ')
        result = pytester.runpytest()
        assert result.ret != 0
        assert 'should be seen' in result.stdout.str()

    def test_issue109_sibling_conftests_not_loaded(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        sub1 = pytester.mkdir('sub1')
        sub2 = pytester.mkdir('sub2')
        sub1.joinpath('conftest.py').write_text('assert 0', encoding='utf-8')
        result = pytester.runpytest(sub2)
        assert result.ret == ExitCode.NO_TESTS_COLLECTED
        sub2.joinpath('__init__.py').touch()
        p = sub2.joinpath('test_hello.py')
        p.touch()
        result = pytester.runpytest(p)
        assert result.ret == ExitCode.NO_TESTS_COLLECTED
        result = pytester.runpytest(sub1)
        assert result.ret == ExitCode.USAGE_ERROR

    def test_directory_skipped(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makeconftest('\n            import pytest\n            def pytest_ignore_collect():\n                pytest.skip("intentional")\n        ')
        pytester.makepyfile('def test_hello(): pass')
        result = pytester.runpytest()
        assert result.ret == ExitCode.NO_TESTS_COLLECTED
        result.stdout.fnmatch_lines(['*1 skipped*'])

    def test_multiple_items_per_collector_byid(self, pytester: Pytester) -> None:
        if False:
            return 10
        c = pytester.makeconftest('\n            import pytest\n            class MyItem(pytest.Item):\n                def runtest(self):\n                    pass\n            class MyCollector(pytest.File):\n                def collect(self):\n                    return [MyItem.from_parent(name="xyz", parent=self)]\n            def pytest_collect_file(file_path, parent):\n                if file_path.name.startswith("conftest"):\n                    return MyCollector.from_parent(path=file_path, parent=parent)\n        ')
        result = pytester.runpytest(c.name + '::' + 'xyz')
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*1 pass*'])

    def test_skip_on_generated_funcarg_id(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makeconftest('\n            import pytest\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize(\'x\', [3], ids=[\'hello-123\'])\n            def pytest_runtest_setup(item):\n                print(item.keywords)\n                if \'hello-123\' in item.keywords:\n                    pytest.skip("hello")\n                assert 0\n        ')
        p = pytester.makepyfile('def test_func(x): pass')
        res = pytester.runpytest(p)
        assert res.ret == 0
        res.stdout.fnmatch_lines(['*1 skipped*'])

    def test_direct_addressing_selects(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p = pytester.makepyfile('\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize(\'i\', [1, 2], ids=["1", "2"])\n            def test_func(i):\n                pass\n        ')
        res = pytester.runpytest(p.name + '::' + 'test_func[1]')
        assert res.ret == 0
        res.stdout.fnmatch_lines(['*1 passed*'])

    def test_direct_addressing_selects_duplicates(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p = pytester.makepyfile('\n            import pytest\n\n            @pytest.mark.parametrize("a", [1, 2, 10, 11, 2, 1, 12, 11])\n            def test_func(a):\n                pass\n            ')
        result = pytester.runpytest(p)
        result.assert_outcomes(failed=0, passed=8)

    def test_direct_addressing_selects_duplicates_1(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p = pytester.makepyfile('\n            import pytest\n\n            @pytest.mark.parametrize("a", [1, 2, 10, 11, 2, 1, 12, 1_1,2_1])\n            def test_func(a):\n                pass\n            ')
        result = pytester.runpytest(p)
        result.assert_outcomes(failed=0, passed=9)

    def test_direct_addressing_selects_duplicates_2(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = pytester.makepyfile('\n            import pytest\n\n            @pytest.mark.parametrize("a", ["a","b","c","a","a1"])\n            def test_func(a):\n                pass\n            ')
        result = pytester.runpytest(p)
        result.assert_outcomes(failed=0, passed=5)

    def test_direct_addressing_notfound(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = pytester.makepyfile('\n            def test_func():\n                pass\n        ')
        res = pytester.runpytest(p.name + '::' + 'test_notfound')
        assert res.ret
        res.stderr.fnmatch_lines(['*ERROR*not found*'])

    def test_docstring_on_hookspec(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        from _pytest import hookspec
        for (name, value) in vars(hookspec).items():
            if name.startswith('pytest_'):
                assert value.__doc__, 'no docstring for %s' % name

    def test_initialization_error_issue49(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makeconftest('\n            def pytest_configure():\n                x\n        ')
        result = pytester.runpytest()
        assert result.ret == 3
        result.stderr.fnmatch_lines(['INTERNAL*pytest_configure*', 'INTERNAL*x*'])
        assert 'sessionstarttime' not in result.stderr.str()

    @pytest.mark.parametrize('lookfor', ['test_fun.py::test_a'])
    def test_issue134_report_error_when_collecting_member(self, pytester: Pytester, lookfor) -> None:
        if False:
            return 10
        pytester.makepyfile(test_fun='\n            def test_a():\n                pass\n            def')
        result = pytester.runpytest(lookfor)
        result.stdout.fnmatch_lines(['*SyntaxError*'])
        if '::' in lookfor:
            result.stderr.fnmatch_lines(['*ERROR*'])
            assert result.ret == 4

    def test_report_all_failed_collections_initargs(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makeconftest('\n            from _pytest.config import ExitCode\n\n            def pytest_sessionfinish(exitstatus):\n                assert exitstatus == ExitCode.USAGE_ERROR\n                print("pytest_sessionfinish_called")\n            ')
        pytester.makepyfile(test_a='def', test_b='def')
        result = pytester.runpytest('test_a.py::a', 'test_b.py::b')
        result.stderr.fnmatch_lines(['*ERROR*test_a.py::a*', '*ERROR*test_b.py::b*'])
        result.stdout.fnmatch_lines(['pytest_sessionfinish_called'])
        assert result.ret == ExitCode.USAGE_ERROR

    def test_namespace_import_doesnt_confuse_import_hook(self, pytester: Pytester) -> None:
        if False:
            return 10
        "Ref #383.\n\n        Python 3.3's namespace package messed with our import hooks.\n        Importing a module that didn't exist, even if the ImportError was\n        gracefully handled, would make our test crash.\n        "
        pytester.mkdir('not_a_package')
        p = pytester.makepyfile('\n            try:\n                from not_a_package import doesnt_exist\n            except ImportError:\n                # We handle the import error gracefully here\n                pass\n\n            def test_whatever():\n                pass\n        ')
        res = pytester.runpytest(p.name)
        assert res.ret == 0

    def test_unknown_option(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        result = pytester.runpytest('--qwlkej')
        result.stderr.fnmatch_lines('\n            *unrecognized*\n        ')

    def test_getsourcelines_error_issue553(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            print('Hello World!')
        monkeypatch.setattr('inspect.getsourcelines', None)
        p = pytester.makepyfile("\n            def raise_error(obj):\n                raise OSError('source code not available')\n\n            import inspect\n            inspect.getsourcelines = raise_error\n\n            def test_foo(invalid_fixture):\n                pass\n        ")
        res = pytester.runpytest(p)
        res.stdout.fnmatch_lines(['*source code not available*', "E*fixture 'invalid_fixture' not found"])

    def test_plugins_given_as_strings(self, pytester: Pytester, monkeypatch, _sys_snapshot) -> None:
        if False:
            while True:
                i = 10
        'Test that str values passed to main() as `plugins` arg are\n        interpreted as module names to be imported and registered (#855).'
        with pytest.raises(ImportError) as excinfo:
            pytest.main([str(pytester.path)], plugins=['invalid.module'])
        assert 'invalid' in str(excinfo.value)
        p = pytester.path.joinpath('test_test_plugins_given_as_strings.py')
        p.write_text('def test_foo(): pass', encoding='utf-8')
        mod = types.ModuleType('myplugin')
        monkeypatch.setitem(sys.modules, 'myplugin', mod)
        assert pytest.main(args=[str(pytester.path)], plugins=['myplugin']) == 0

    def test_parametrized_with_bytes_regex(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p = pytester.makepyfile("\n            import re\n            import pytest\n            @pytest.mark.parametrize('r', [re.compile(b'foo')])\n            def test_stuff(r):\n                pass\n        ")
        res = pytester.runpytest(p)
        res.stdout.fnmatch_lines(['*1 passed*'])

    def test_parametrized_with_null_bytes(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test parametrization with values that contain null bytes and unicode characters (#2644, #2957)'
        p = pytester.makepyfile('            import pytest\n\n            @pytest.mark.parametrize("data", [b"\\x00", "\\x00", \'ação\'])\n            def test_foo(data):\n                assert data\n            ')
        res = pytester.runpytest(p)
        res.assert_outcomes(passed=3)

class TestInvocationVariants:

    def test_earlyinit(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p = pytester.makepyfile("\n            import pytest\n            assert hasattr(pytest, 'mark')\n        ")
        result = pytester.runpython(p)
        assert result.ret == 0

    def test_pydoc(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        result = pytester.runpython_c('import pytest;help(pytest)')
        assert result.ret == 0
        s = result.stdout.str()
        assert 'MarkGenerator' in s

    def test_import_star_pytest(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p = pytester.makepyfile('\n            from pytest import *\n            #Item\n            #File\n            main\n            skip\n            xfail\n        ')
        result = pytester.runpython(p)
        assert result.ret == 0

    def test_double_pytestcmdline(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p = pytester.makepyfile(run='\n            import pytest\n            pytest.main()\n            pytest.main()\n        ')
        pytester.makepyfile('\n            def test_hello():\n                pass\n        ')
        result = pytester.runpython(p)
        result.stdout.fnmatch_lines(['*1 passed*', '*1 passed*'])

    def test_python_minus_m_invocation_ok(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p1 = pytester.makepyfile('def test_hello(): pass')
        res = pytester.run(sys.executable, '-m', 'pytest', str(p1))
        assert res.ret == 0

    def test_python_minus_m_invocation_fail(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p1 = pytester.makepyfile('def test_fail(): 0/0')
        res = pytester.run(sys.executable, '-m', 'pytest', str(p1))
        assert res.ret == 1

    def test_python_pytest_package(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p1 = pytester.makepyfile('def test_pass(): pass')
        res = pytester.run(sys.executable, '-m', 'pytest', str(p1))
        assert res.ret == 0
        res.stdout.fnmatch_lines(['*1 passed*'])

    def test_invoke_with_invalid_type(self) -> None:
        if False:
            return 10
        with pytest.raises(TypeError, match="expected to be a list of strings, got: '-h'"):
            pytest.main('-h')

    def test_invoke_with_path(self, pytester: Pytester, capsys) -> None:
        if False:
            print('Hello World!')
        retcode = pytest.main([str(pytester.path)])
        assert retcode == ExitCode.NO_TESTS_COLLECTED
        (out, err) = capsys.readouterr()

    def test_invoke_plugin_api(self, capsys) -> None:
        if False:
            return 10

        class MyPlugin:

            def pytest_addoption(self, parser):
                if False:
                    for i in range(10):
                        print('nop')
                parser.addoption('--myopt')
        pytest.main(['-h'], plugins=[MyPlugin()])
        (out, err) = capsys.readouterr()
        assert '--myopt' in out

    def test_pyargs_importerror(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            i = 10
            return i + 15
        monkeypatch.delenv('PYTHONDONTWRITEBYTECODE', False)
        path = pytester.mkpydir('tpkg')
        path.joinpath('test_hello.py').write_text('raise ImportError', encoding='utf-8')
        result = pytester.runpytest('--pyargs', 'tpkg.test_hello', syspathinsert=True)
        assert result.ret != 0
        result.stdout.fnmatch_lines(['collected*0*items*/*1*error'])

    def test_pyargs_only_imported_once(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pkg = pytester.mkpydir('foo')
        pkg.joinpath('test_foo.py').write_text("print('hello from test_foo')\ndef test(): pass", encoding='utf-8')
        pkg.joinpath('conftest.py').write_text("def pytest_configure(config): print('configuring')", encoding='utf-8')
        result = pytester.runpytest('--pyargs', 'foo.test_foo', '-s', syspathinsert=True)
        assert result.outlines.count('hello from test_foo') == 1
        assert result.outlines.count('configuring') == 1

    def test_pyargs_filename_looks_like_module(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.path.joinpath('conftest.py').touch()
        pytester.path.joinpath('t.py').write_text('def test(): pass', encoding='utf-8')
        result = pytester.runpytest('--pyargs', 't.py')
        assert result.ret == ExitCode.OK

    def test_cmdline_python_package(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            for i in range(10):
                print('nop')
        import warnings
        monkeypatch.delenv('PYTHONDONTWRITEBYTECODE', False)
        path = pytester.mkpydir('tpkg')
        path.joinpath('test_hello.py').write_text('def test_hello(): pass', encoding='utf-8')
        path.joinpath('test_world.py').write_text('def test_world(): pass', encoding='utf-8')
        result = pytester.runpytest('--pyargs', 'tpkg')
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*2 passed*'])
        result = pytester.runpytest('--pyargs', 'tpkg.test_hello', syspathinsert=True)
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*1 passed*'])
        empty_package = pytester.mkpydir('empty_package')
        monkeypatch.setenv('PYTHONPATH', str(empty_package), prepend=os.pathsep)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ImportWarning)
            result = pytester.runpytest('--pyargs', '.')
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*2 passed*'])
        monkeypatch.setenv('PYTHONPATH', str(pytester), prepend=os.pathsep)
        result = pytester.runpytest('--pyargs', 'tpkg.test_missing', syspathinsert=True)
        assert result.ret != 0
        result.stderr.fnmatch_lines(['*not*found*test_missing*'])

    def test_cmdline_python_namespace_package(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            while True:
                i = 10
        'Test --pyargs option with namespace packages (#1567).\n\n        Ref: https://packaging.python.org/guides/packaging-namespace-packages/\n        '
        monkeypatch.delenv('PYTHONDONTWRITEBYTECODE', raising=False)
        search_path = []
        for dirname in ('hello', 'world'):
            d = pytester.mkdir(dirname)
            search_path.append(d)
            ns = d.joinpath('ns_pkg')
            ns.mkdir()
            ns.joinpath('__init__.py').write_text("__import__('pkg_resources').declare_namespace(__name__)", encoding='utf-8')
            lib = ns.joinpath(dirname)
            lib.mkdir()
            lib.joinpath('__init__.py').touch()
            lib.joinpath(f'test_{dirname}.py').write_text(f'def test_{dirname}(): pass\ndef test_other():pass', encoding='utf-8')
        monkeypatch.setenv('PYTHONPATH', prepend_pythonpath(*search_path))
        for p in search_path:
            monkeypatch.syspath_prepend(p)
        monkeypatch.chdir('world')
        ignore_w = ('-Wignore:Deprecated call to `pkg_resources.declare_namespace', '-Wignore:pkg_resources is deprecated')
        result = pytester.runpytest('--pyargs', '-v', 'ns_pkg.hello', 'ns_pkg/world', *ignore_w)
        assert result.ret == 0
        result.stdout.fnmatch_lines(['test_hello.py::test_hello*PASSED*', 'test_hello.py::test_other*PASSED*', 'ns_pkg/world/test_world.py::test_world*PASSED*', 'ns_pkg/world/test_world.py::test_other*PASSED*', '*4 passed in*'])
        pytester.chdir()
        result = pytester.runpytest('--pyargs', '-v', 'ns_pkg.world.test_world::test_other')
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*test_world.py::test_other*PASSED*', '*1 passed*'])

    def test_invoke_test_and_doctestmodules(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p = pytester.makepyfile('\n            def test():\n                pass\n        ')
        result = pytester.runpytest(str(p) + '::test', '--doctest-modules')
        result.stdout.fnmatch_lines(['*1 passed*'])

    def test_cmdline_python_package_symlink(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            while True:
                i = 10
        '\n        --pyargs with packages with path containing symlink can have conftest.py in\n        their package (#2985)\n        '
        monkeypatch.delenv('PYTHONDONTWRITEBYTECODE', raising=False)
        dirname = 'lib'
        d = pytester.mkdir(dirname)
        foo = d.joinpath('foo')
        foo.mkdir()
        foo.joinpath('__init__.py').touch()
        lib = foo.joinpath('bar')
        lib.mkdir()
        lib.joinpath('__init__.py').touch()
        lib.joinpath('test_bar.py').write_text('def test_bar(): pass\ndef test_other(a_fixture):pass', encoding='utf-8')
        lib.joinpath('conftest.py').write_text('import pytest\n@pytest.fixture\ndef a_fixture():pass', encoding='utf-8')
        d_local = pytester.mkdir('symlink_root')
        symlink_location = d_local / 'lib'
        symlink_or_skip(d, symlink_location, target_is_directory=True)
        search_path = ['lib', os.path.join('symlink_root', 'lib')]
        monkeypatch.setenv('PYTHONPATH', prepend_pythonpath(*search_path))
        for p in search_path:
            monkeypatch.syspath_prepend(p)
        result = pytester.runpytest('--pyargs', '-v', 'foo.bar')
        pytester.chdir()
        assert result.ret == 0
        result.stdout.fnmatch_lines(['symlink_root/lib/foo/bar/test_bar.py::test_bar PASSED*', 'symlink_root/lib/foo/bar/test_bar.py::test_other PASSED*', '*2 passed*'])

    def test_cmdline_python_package_not_exists(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        result = pytester.runpytest('--pyargs', 'tpkgwhatv')
        assert result.ret
        result.stderr.fnmatch_lines(['ERROR*module*or*package*not*found*'])

    @pytest.mark.xfail(reason='decide: feature or bug')
    def test_noclass_discovery_if_not_testcase(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        testpath = pytester.makepyfile('\n            import unittest\n            class TestHello(object):\n                def test_hello(self):\n                    assert self.attr\n\n            class RealTest(unittest.TestCase, TestHello):\n                attr = 42\n        ')
        reprec = pytester.inline_run(testpath)
        reprec.assertoutcome(passed=1)

    def test_doctest_id(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makefile('.txt', '\n            >>> x=3\n            >>> x\n            4\n        ')
        testid = 'test_doctest_id.txt::test_doctest_id.txt'
        expected_lines = ['*= FAILURES =*', '*_ ?doctest? test_doctest_id.txt _*', 'FAILED test_doctest_id.txt::test_doctest_id.txt', '*= 1 failed in*']
        result = pytester.runpytest(testid, '-rf', '--tb=short')
        result.stdout.fnmatch_lines(expected_lines)
        result = pytester.runpytest(testid, '-rf', '--tb=short')
        result.stdout.fnmatch_lines(expected_lines)

    def test_core_backward_compatibility(self) -> None:
        if False:
            return 10
        'Test backward compatibility for get_plugin_manager function. See #787.'
        import _pytest.config
        assert type(_pytest.config.get_plugin_manager()) is _pytest.config.PytestPluginManager

    def test_has_plugin(self, request) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test hasplugin function of the plugin manager (#932).'
        assert request.config.pluginmanager.hasplugin('python')

class TestDurations:
    source = '\n        from _pytest import timing\n        def test_something():\n            pass\n        def test_2():\n            timing.sleep(0.010)\n        def test_1():\n            timing.sleep(0.002)\n        def test_3():\n            timing.sleep(0.020)\n    '

    def test_calls(self, pytester: Pytester, mock_timing) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile(self.source)
        result = pytester.runpytest_inprocess('--durations=10')
        assert result.ret == 0
        result.stdout.fnmatch_lines_random(['*durations*', '*call*test_3*', '*call*test_2*'])
        result.stdout.fnmatch_lines(['(8 durations < 0.005s hidden.  Use -vv to show these durations.)'])

    def test_calls_show_2(self, pytester: Pytester, mock_timing) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile(self.source)
        result = pytester.runpytest_inprocess('--durations=2')
        assert result.ret == 0
        lines = result.stdout.get_lines_after('*slowest*durations*')
        assert '4 passed' in lines[2]

    def test_calls_showall(self, pytester: Pytester, mock_timing) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile(self.source)
        result = pytester.runpytest_inprocess('--durations=0')
        assert result.ret == 0
        tested = '3'
        for x in tested:
            for y in ('call',):
                for line in result.stdout.lines:
                    if 'test_%s' % x in line and y in line:
                        break
                else:
                    raise AssertionError(f'not found {x} {y}')

    def test_calls_showall_verbose(self, pytester: Pytester, mock_timing) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile(self.source)
        result = pytester.runpytest_inprocess('--durations=0', '-vv')
        assert result.ret == 0
        for x in '123':
            for y in ('call',):
                for line in result.stdout.lines:
                    if 'test_%s' % x in line and y in line:
                        break
                else:
                    raise AssertionError(f'not found {x} {y}')

    def test_with_deselected(self, pytester: Pytester, mock_timing) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile(self.source)
        result = pytester.runpytest_inprocess('--durations=2', '-k test_3')
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*durations*', '*call*test_3*'])

    def test_with_failing_collection(self, pytester: Pytester, mock_timing) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile(self.source)
        pytester.makepyfile(test_collecterror='xyz')
        result = pytester.runpytest_inprocess('--durations=2', '-k test_1')
        assert result.ret == 2
        result.stdout.fnmatch_lines(['*Interrupted: 1 error during collection*'])
        result.stdout.no_fnmatch_line('*duration*')

    def test_with_not(self, pytester: Pytester, mock_timing) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile(self.source)
        result = pytester.runpytest_inprocess('-k not 1')
        assert result.ret == 0

class TestDurationsWithFixture:
    source = '\n        import pytest\n        from _pytest import timing\n\n        @pytest.fixture\n        def setup_fixt():\n            timing.sleep(2)\n\n        def test_1(setup_fixt):\n            timing.sleep(5)\n    '

    def test_setup_function(self, pytester: Pytester, mock_timing) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile(self.source)
        result = pytester.runpytest_inprocess('--durations=10')
        assert result.ret == 0
        result.stdout.fnmatch_lines_random('\n            *durations*\n            5.00s call *test_1*\n            2.00s setup *test_1*\n        ')

def test_zipimport_hook(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    'Test package loader is being used correctly (see #1837).'
    zipapp = pytest.importorskip('zipapp')
    pytester.path.joinpath('app').mkdir()
    pytester.makepyfile(**{'app/foo.py': "\n            import pytest\n            def main():\n                pytest.main(['--pyargs', 'foo'])\n        "})
    target = pytester.path.joinpath('foo.zip')
    zipapp.create_archive(str(pytester.path.joinpath('app')), str(target), main='foo:main')
    result = pytester.runpython(target)
    assert result.ret == 0
    result.stderr.fnmatch_lines(['*not found*foo*'])
    result.stdout.no_fnmatch_line('*INTERNALERROR>*')

def test_import_plugin_unicode_name(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile(myplugin='')
    pytester.makepyfile('def test(): pass')
    pytester.makeconftest("pytest_plugins = ['myplugin']")
    r = pytester.runpytest()
    assert r.ret == 0

def test_pytest_plugins_as_module(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    'Do not raise an error if pytest_plugins attribute is a module (#3899)'
    pytester.makepyfile(**{'__init__.py': '', 'pytest_plugins.py': '', 'conftest.py': 'from . import pytest_plugins', 'test_foo.py': 'def test(): pass'})
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['* 1 passed in *'])

def test_deferred_hook_checking(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    'Check hooks as late as possible (#1821).'
    pytester.syspathinsert()
    pytester.makepyfile(**{'plugin.py': '\n        class Hooks(object):\n            def pytest_my_hook(self, config):\n                pass\n\n        def pytest_configure(config):\n            config.pluginmanager.add_hookspecs(Hooks)\n        ', 'conftest.py': "\n            pytest_plugins = ['plugin']\n            def pytest_my_hook(config):\n                return 40\n        ", 'test_foo.py': '\n            def test(request):\n                assert request.config.hook.pytest_my_hook(config=request.config) == [40]\n        '})
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['* 1 passed *'])

def test_fixture_values_leak(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    'Ensure that fixture objects are properly destroyed by the garbage collector at the end of their expected\n    life-times (#2981).\n    '
    pytester.makepyfile('\n        import dataclasses\n        import gc\n        import pytest\n        import weakref\n\n        @dataclasses.dataclass\n        class SomeObj:\n            name: str\n\n        fix_of_test1_ref = None\n        session_ref = None\n\n        @pytest.fixture(scope=\'session\')\n        def session_fix():\n            global session_ref\n            obj = SomeObj(name=\'session-fixture\')\n            session_ref = weakref.ref(obj)\n            return obj\n\n        @pytest.fixture\n        def fix(session_fix):\n            global fix_of_test1_ref\n            obj = SomeObj(name=\'local-fixture\')\n            fix_of_test1_ref = weakref.ref(obj)\n            return obj\n\n        def test1(fix):\n            assert fix_of_test1_ref() is fix\n\n        def test2():\n            gc.collect()\n            # fixture "fix" created during test1 must have been destroyed by now\n            assert fix_of_test1_ref() is None\n    ')
    result = pytester.runpytest_subprocess()
    result.stdout.fnmatch_lines(['* 2 passed *'])

def test_fixture_order_respects_scope(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    'Ensure that fixtures are created according to scope order (#2405).'
    pytester.makepyfile("\n        import pytest\n\n        data = {}\n\n        @pytest.fixture(scope='module')\n        def clean_data():\n            data.clear()\n\n        @pytest.fixture(autouse=True)\n        def add_data():\n            data.update(value=True)\n\n        @pytest.mark.usefixtures('clean_data')\n        def test_value():\n            assert data.get('value')\n    ")
    result = pytester.runpytest()
    assert result.ret == 0

def test_frame_leak_on_failing_test(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    'Pytest would leak garbage referencing the frames of tests that failed\n    that could never be reclaimed (#2798).\n\n    Unfortunately it was not possible to remove the actual circles because most of them\n    are made of traceback objects which cannot be weakly referenced. Those objects at least\n    can be eventually claimed by the garbage collector.\n    '
    pytester.makepyfile('\n        import gc\n        import weakref\n\n        class Obj:\n            pass\n\n        ref = None\n\n        def test1():\n            obj = Obj()\n            global ref\n            ref = weakref.ref(obj)\n            assert 0\n\n        def test2():\n            gc.collect()\n            assert ref() is None\n    ')
    result = pytester.runpytest_subprocess()
    result.stdout.fnmatch_lines(['*1 failed, 1 passed in*'])

def test_fixture_mock_integration(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    'Test that decorators applied to fixture are left working (#3774)'
    p = pytester.copy_example('acceptance/fixture_mock_integration.py')
    result = pytester.runpytest(p)
    result.stdout.fnmatch_lines(['*1 passed*'])

def test_usage_error_code(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    result = pytester.runpytest('-unknown-option-')
    assert result.ret == ExitCode.USAGE_ERROR

def test_warn_on_async_function(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile(test_async='\n        async def test_1():\n            pass\n        async def test_2():\n            pass\n        def test_3():\n            coro = test_2()\n            coro.close()\n            return coro\n    ')
    result = pytester.runpytest('-Wdefault')
    result.stdout.fnmatch_lines(['test_async.py::test_1', 'test_async.py::test_2', 'test_async.py::test_3', '*async def functions are not natively supported*', '*3 skipped, 3 warnings in*'])
    assert result.stdout.str().count('async def functions are not natively supported') == 1

def test_warn_on_async_gen_function(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile(test_async='\n        async def test_1():\n            yield\n        async def test_2():\n            yield\n        def test_3():\n            return test_2()\n    ')
    result = pytester.runpytest('-Wdefault')
    result.stdout.fnmatch_lines(['test_async.py::test_1', 'test_async.py::test_2', 'test_async.py::test_3', '*async def functions are not natively supported*', '*3 skipped, 3 warnings in*'])
    assert result.stdout.str().count('async def functions are not natively supported') == 1

def test_pdb_can_be_rewritten(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile(**{'conftest.py': '\n                import pytest\n                pytest.register_assert_rewrite("pdb")\n                ', '__init__.py': '', 'pdb.py': '\n                def check():\n                    assert 1 == 2\n                ', 'test_pdb.py': '\n                def test():\n                    import pdb\n                    assert pdb.check()\n                '})
    result = pytester.runpytest_subprocess('-p', 'no:debugging', '-vv')
    result.stdout.fnmatch_lines(['    def check():', '>       assert 1 == 2', 'E       assert 1 == 2', '', 'pdb.py:2: AssertionError', '*= 1 failed in *'])
    assert result.ret == 1

def test_tee_stdio_captures_and_live_prints(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    testpath = pytester.makepyfile('\n        import sys\n        def test_simple():\n            print ("@this is stdout@")\n            print ("@this is stderr@", file=sys.stderr)\n    ')
    result = pytester.runpytest_subprocess(testpath, '--capture=tee-sys', '--junitxml=output.xml', '-o', 'junit_logging=all')
    result.stdout.fnmatch_lines(['*@this is stdout@*'])
    result.stderr.fnmatch_lines(['*@this is stderr@*'])
    fullXml = pytester.path.joinpath('output.xml').read_text(encoding='utf-8')
    assert '@this is stdout@\n' in fullXml
    assert '@this is stderr@\n' in fullXml

@pytest.mark.skipif(sys.platform == 'win32', reason='Windows raises `OSError: [Errno 22] Invalid argument` instead')
def test_no_brokenpipeerror_message(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    'Ensure that the broken pipe error message is suppressed.\n\n    In some Python versions, it reaches sys.unraisablehook, in others\n    a BrokenPipeError exception is propagated, but either way it prints\n    to stderr on shutdown, so checking nothing is printed is enough.\n    '
    popen = pytester.popen((*pytester._getpytestargs(), '--help'))
    popen.stdout.close()
    ret = popen.wait()
    assert popen.stderr.read() == b''
    assert ret == 1
    popen.stderr.close()

def test_function_return_non_none_warning(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile('\n        def test_stuff():\n            return "something"\n    ')
    res = pytester.runpytest()
    res.stdout.fnmatch_lines(['*Did you mean to use `assert` instead of `return`?*'])

def test_doctest_and_normal_imports_with_importlib(pytester: Pytester) -> None:
    if False:
        return 10
    '\n    Regression test for #10811: previously import_path with ImportMode.importlib would\n    not return a module if already in sys.modules, resulting in modules being imported\n    multiple times, which causes problems with modules that have import side effects.\n    '
    pytester.makepyfile(**{'pmxbot/commands.py': 'from . import logging', 'pmxbot/logging.py': '', 'tests/__init__.py': '', 'tests/test_commands.py': "\n                import importlib\n                from pmxbot import logging\n\n                class TestCommands:\n                    def test_boo(self):\n                        assert importlib.import_module('pmxbot.logging') is logging\n                "})
    pytester.makeini('\n        [pytest]\n        addopts=\n            --doctest-modules\n            --import-mode importlib\n        ')
    result = pytester.runpytest_subprocess()
    result.stdout.fnmatch_lines('*1 passed*')