import os
import textwrap
from pathlib import Path
from typing import cast
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
import pytest
from _pytest.config import ExitCode
from _pytest.config import PytestPluginManager
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pathlib import symlink_or_skip
from _pytest.pytester import Pytester
from _pytest.tmpdir import TempPathFactory

def ConftestWithSetinitial(path) -> PytestPluginManager:
    if False:
        return 10
    conftest = PytestPluginManager()
    conftest_setinitial(conftest, [path])
    return conftest

def conftest_setinitial(conftest: PytestPluginManager, args: Sequence[Union[str, Path]], confcutdir: Optional[Path]=None) -> None:
    if False:
        return 10
    conftest._set_initial_conftests(args=args, pyargs=False, noconftest=False, rootpath=Path(args[0]), confcutdir=confcutdir, importmode='prepend')

@pytest.mark.usefixtures('_sys_snapshot')
class TestConftestValueAccessGlobal:

    @pytest.fixture(scope='module', params=['global', 'inpackage'])
    def basedir(self, request, tmp_path_factory: TempPathFactory) -> Generator[Path, None, None]:
        if False:
            i = 10
            return i + 15
        tmp_path = tmp_path_factory.mktemp('basedir', numbered=True)
        tmp_path.joinpath('adir/b').mkdir(parents=True)
        tmp_path.joinpath('adir/conftest.py').write_text('a=1 ; Directory = 3', encoding='utf-8')
        tmp_path.joinpath('adir/b/conftest.py').write_text('b=2 ; a = 1.5', encoding='utf-8')
        if request.param == 'inpackage':
            tmp_path.joinpath('adir/__init__.py').touch()
            tmp_path.joinpath('adir/b/__init__.py').touch()
        yield tmp_path

    def test_basic_init(self, basedir: Path) -> None:
        if False:
            i = 10
            return i + 15
        conftest = PytestPluginManager()
        p = basedir / 'adir'
        conftest._loadconftestmodules(p, importmode='prepend', rootpath=basedir)
        assert conftest._rget_with_confmod('a', p)[1] == 1

    def test_immediate_initialiation_and_incremental_are_the_same(self, basedir: Path) -> None:
        if False:
            return 10
        conftest = PytestPluginManager()
        assert not len(conftest._dirpath2confmods)
        conftest._loadconftestmodules(basedir, importmode='prepend', rootpath=basedir)
        snap1 = len(conftest._dirpath2confmods)
        assert snap1 == 1
        conftest._loadconftestmodules(basedir / 'adir', importmode='prepend', rootpath=basedir)
        assert len(conftest._dirpath2confmods) == snap1 + 1
        conftest._loadconftestmodules(basedir / 'b', importmode='prepend', rootpath=basedir)
        assert len(conftest._dirpath2confmods) == snap1 + 2

    def test_value_access_not_existing(self, basedir: Path) -> None:
        if False:
            return 10
        conftest = ConftestWithSetinitial(basedir)
        with pytest.raises(KeyError):
            conftest._rget_with_confmod('a', basedir)

    def test_value_access_by_path(self, basedir: Path) -> None:
        if False:
            i = 10
            return i + 15
        conftest = ConftestWithSetinitial(basedir)
        adir = basedir / 'adir'
        conftest._loadconftestmodules(adir, importmode='prepend', rootpath=basedir)
        assert conftest._rget_with_confmod('a', adir)[1] == 1
        conftest._loadconftestmodules(adir / 'b', importmode='prepend', rootpath=basedir)
        assert conftest._rget_with_confmod('a', adir / 'b')[1] == 1.5

    def test_value_access_with_confmod(self, basedir: Path) -> None:
        if False:
            i = 10
            return i + 15
        startdir = basedir / 'adir' / 'b'
        startdir.joinpath('xx').mkdir()
        conftest = ConftestWithSetinitial(startdir)
        (mod, value) = conftest._rget_with_confmod('a', startdir)
        assert value == 1.5
        assert mod.__file__ is not None
        path = Path(mod.__file__)
        assert path.parent == basedir / 'adir' / 'b'
        assert path.stem == 'conftest'

def test_conftest_in_nonpkg_with_init(tmp_path: Path, _sys_snapshot) -> None:
    if False:
        print('Hello World!')
    tmp_path.joinpath('adir-1.0/b').mkdir(parents=True)
    tmp_path.joinpath('adir-1.0/conftest.py').write_text('a=1 ; Directory = 3', encoding='utf-8')
    tmp_path.joinpath('adir-1.0/b/conftest.py').write_text('b=2 ; a = 1.5', encoding='utf-8')
    tmp_path.joinpath('adir-1.0/b/__init__.py').touch()
    tmp_path.joinpath('adir-1.0/__init__.py').touch()
    ConftestWithSetinitial(tmp_path.joinpath('adir-1.0', 'b'))

def test_doubledash_considered(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    conf = pytester.mkdir('--option')
    conf.joinpath('conftest.py').touch()
    conftest = PytestPluginManager()
    conftest_setinitial(conftest, [conf.name, conf.name])
    values = conftest._getconftestmodules(conf)
    assert len(values) == 1

def test_issue151_load_all_conftests(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    names = 'code proj src'.split()
    for name in names:
        p = pytester.mkdir(name)
        p.joinpath('conftest.py').touch()
    pm = PytestPluginManager()
    conftest_setinitial(pm, names)
    assert len(set(pm.get_plugins()) - {pm}) == len(names)

def test_conftest_global_import(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makeconftest('x=3')
    p = pytester.makepyfile('\n        from pathlib import Path\n        import pytest\n        from _pytest.config import PytestPluginManager\n        conf = PytestPluginManager()\n        mod = conf._importconftest(Path("conftest.py"), importmode="prepend", rootpath=Path.cwd())\n        assert mod.x == 3\n        import conftest\n        assert conftest is mod, (conftest, mod)\n        sub = Path("sub")\n        sub.mkdir()\n        subconf = sub / "conftest.py"\n        subconf.write_text("y=4", encoding="utf-8")\n        mod2 = conf._importconftest(subconf, importmode="prepend", rootpath=Path.cwd())\n        assert mod != mod2\n        assert mod2.y == 4\n        import conftest\n        assert conftest is mod2, (conftest, mod)\n    ')
    res = pytester.runpython(p)
    assert res.ret == 0

def test_conftestcutdir(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    conf = pytester.makeconftest('')
    p = pytester.mkdir('x')
    conftest = PytestPluginManager()
    conftest_setinitial(conftest, [pytester.path], confcutdir=p)
    conftest._loadconftestmodules(p, importmode='prepend', rootpath=pytester.path)
    values = conftest._getconftestmodules(p)
    assert len(values) == 0
    conftest._loadconftestmodules(conf.parent, importmode='prepend', rootpath=pytester.path)
    values = conftest._getconftestmodules(conf.parent)
    assert len(values) == 0
    assert not conftest.has_plugin(str(conf))
    conftest._importconftest(conf, importmode='prepend', rootpath=pytester.path)
    values = conftest._getconftestmodules(conf.parent)
    assert values[0].__file__ is not None
    assert values[0].__file__.startswith(str(conf))
    values = conftest._getconftestmodules(p)
    assert len(values) == 1
    assert values[0].__file__ is not None
    assert values[0].__file__.startswith(str(conf))

def test_conftestcutdir_inplace_considered(pytester: Pytester) -> None:
    if False:
        return 10
    conf = pytester.makeconftest('')
    conftest = PytestPluginManager()
    conftest_setinitial(conftest, [conf.parent], confcutdir=conf.parent)
    values = conftest._getconftestmodules(conf.parent)
    assert len(values) == 1
    assert values[0].__file__ is not None
    assert values[0].__file__.startswith(str(conf))

@pytest.mark.parametrize('name', 'test tests whatever .dotdir'.split())
def test_setinitial_conftest_subdirs(pytester: Pytester, name: str) -> None:
    if False:
        print('Hello World!')
    sub = pytester.mkdir(name)
    subconftest = sub.joinpath('conftest.py')
    subconftest.touch()
    pm = PytestPluginManager()
    conftest_setinitial(pm, [sub.parent], confcutdir=pytester.path)
    key = subconftest.resolve()
    if name not in ('whatever', '.dotdir'):
        assert pm.has_plugin(str(key))
        assert len(set(pm.get_plugins()) - {pm}) == 1
    else:
        assert not pm.has_plugin(str(key))
        assert len(set(pm.get_plugins()) - {pm}) == 0

def test_conftest_confcutdir(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makeconftest('assert 0')
    x = pytester.mkdir('x')
    x.joinpath('conftest.py').write_text(textwrap.dedent('            def pytest_addoption(parser):\n                parser.addoption("--xyz", action="store_true")\n            '), encoding='utf-8')
    result = pytester.runpytest('-h', '--confcutdir=%s' % x, x)
    result.stdout.fnmatch_lines(['*--xyz*'])
    result.stdout.no_fnmatch_line('*warning: could not load initial*')

def test_installed_conftest_is_picked_up(pytester: Pytester, tmp_path: Path) -> None:
    if False:
        print('Hello World!')
    'When using `--pyargs` to run tests in an installed packages (located e.g.\n    in a site-packages in the PYTHONPATH), conftest files in there are picked\n    up.\n\n    Regression test for #9767.\n    '
    pytester.syspathinsert(tmp_path)
    pytester.makepyprojecttoml('[tool.pytest.ini_options]')
    tmp_path.joinpath('foo').mkdir()
    tmp_path.joinpath('foo', '__init__.py').touch()
    tmp_path.joinpath('foo', 'conftest.py').write_text(textwrap.dedent('            import pytest\n            @pytest.fixture\n            def fix(): return None\n            '), encoding='utf-8')
    tmp_path.joinpath('foo', 'test_it.py').write_text('def test_it(fix): pass', encoding='utf-8')
    result = pytester.runpytest('--pyargs', 'foo')
    assert result.ret == 0

def test_conftest_symlink(pytester: Pytester) -> None:
    if False:
        return 10
    '`conftest.py` discovery follows normal path resolution and does not resolve symlinks.'
    real = pytester.mkdir('real')
    realtests = real.joinpath('app/tests')
    realtests.mkdir(parents=True)
    symlink_or_skip(realtests, pytester.path.joinpath('symlinktests'))
    symlink_or_skip(real, pytester.path.joinpath('symlink'))
    pytester.makepyfile(**{'real/app/tests/test_foo.py': 'def test1(fixture): pass', 'real/conftest.py': textwrap.dedent('\n                import pytest\n\n                print("conftest_loaded")\n\n                @pytest.fixture\n                def fixture():\n                    print("fixture_used")\n                ')})
    result = pytester.runpytest('-vs', 'symlinktests')
    result.stdout.fnmatch_lines(["*fixture 'fixture' not found*"])
    assert result.ret == ExitCode.TESTS_FAILED
    result = pytester.runpytest('-vs', 'symlink')
    assert result.ret == ExitCode.OK

def test_conftest_symlink_files(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    'Symlinked conftest.py are found when pytest is executed in a directory with symlinked\n    files.'
    real = pytester.mkdir('real')
    source = {'app/test_foo.py': 'def test1(fixture): pass', 'app/__init__.py': '', 'app/conftest.py': textwrap.dedent('\n            import pytest\n\n            print("conftest_loaded")\n\n            @pytest.fixture\n            def fixture():\n                print("fixture_used")\n            ')}
    pytester.makepyfile(**{'real/%s' % k: v for (k, v) in source.items()})
    build = pytester.mkdir('build')
    build.joinpath('app').mkdir()
    for f in source:
        symlink_or_skip(real.joinpath(f), build.joinpath(f))
    os.chdir(build)
    result = pytester.runpytest('-vs', 'app/test_foo.py')
    result.stdout.fnmatch_lines(['*conftest_loaded*', 'PASSED'])
    assert result.ret == ExitCode.OK

@pytest.mark.skipif(os.path.normcase('x') != os.path.normcase('X'), reason='only relevant for case insensitive file systems')
def test_conftest_badcase(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    'Check conftest.py loading when directory casing is wrong (#5792).'
    pytester.path.joinpath('JenkinsRoot/test').mkdir(parents=True)
    source = {'setup.py': '', 'test/__init__.py': '', 'test/conftest.py': ''}
    pytester.makepyfile(**{'JenkinsRoot/%s' % k: v for (k, v) in source.items()})
    os.chdir(pytester.path.joinpath('jenkinsroot/test'))
    result = pytester.runpytest()
    assert result.ret == ExitCode.NO_TESTS_COLLECTED

def test_conftest_uppercase(pytester: Pytester) -> None:
    if False:
        return 10
    'Check conftest.py whose qualified name contains uppercase characters (#5819)'
    source = {'__init__.py': '', 'Foo/conftest.py': '', 'Foo/__init__.py': ''}
    pytester.makepyfile(**source)
    os.chdir(pytester.path)
    result = pytester.runpytest()
    assert result.ret == ExitCode.NO_TESTS_COLLECTED

def test_no_conftest(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makeconftest('assert 0')
    result = pytester.runpytest('--noconftest')
    assert result.ret == ExitCode.NO_TESTS_COLLECTED
    result = pytester.runpytest()
    assert result.ret == ExitCode.USAGE_ERROR

def test_conftest_existing_junitxml(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    x = pytester.mkdir('tests')
    x.joinpath('conftest.py').write_text(textwrap.dedent('            def pytest_addoption(parser):\n                parser.addoption("--xyz", action="store_true")\n            '), encoding='utf-8')
    pytester.makefile(ext='.xml', junit='')
    result = pytester.runpytest('-h', '--junitxml', 'junit.xml')
    result.stdout.fnmatch_lines(['*--xyz*'])

def test_conftest_import_order(pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
    if False:
        return 10
    ct1 = pytester.makeconftest('')
    sub = pytester.mkdir('sub')
    ct2 = sub / 'conftest.py'
    ct2.write_text('', encoding='utf-8')

    def impct(p, importmode, root):
        if False:
            return 10
        return p
    conftest = PytestPluginManager()
    conftest._confcutdir = pytester.path
    monkeypatch.setattr(conftest, '_importconftest', impct)
    conftest._loadconftestmodules(sub, importmode='prepend', rootpath=pytester.path)
    mods = cast(List[Path], conftest._getconftestmodules(sub))
    expected = [ct1, ct2]
    assert mods == expected

def test_fixture_dependency(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makeconftest('')
    pytester.path.joinpath('__init__.py').touch()
    sub = pytester.mkdir('sub')
    sub.joinpath('__init__.py').touch()
    sub.joinpath('conftest.py').write_text(textwrap.dedent('            import pytest\n\n            @pytest.fixture\n            def not_needed():\n                assert False, "Should not be called!"\n\n            @pytest.fixture\n            def foo():\n                assert False, "Should not be called!"\n\n            @pytest.fixture\n            def bar(foo):\n                return \'bar\'\n            '), encoding='utf-8')
    subsub = sub.joinpath('subsub')
    subsub.mkdir()
    subsub.joinpath('__init__.py').touch()
    subsub.joinpath('test_bar.py').write_text(textwrap.dedent("            import pytest\n\n            @pytest.fixture\n            def bar():\n                return 'sub bar'\n\n            def test_event_fixture(bar):\n                assert bar == 'sub bar'\n            "), encoding='utf-8')
    result = pytester.runpytest('sub')
    result.stdout.fnmatch_lines(['*1 passed*'])

def test_conftest_found_with_double_dash(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    sub = pytester.mkdir('sub')
    sub.joinpath('conftest.py').write_text(textwrap.dedent('            def pytest_addoption(parser):\n                parser.addoption("--hello-world", action="store_true")\n            '), encoding='utf-8')
    p = sub.joinpath('test_hello.py')
    p.write_text('def test_hello(): pass', encoding='utf-8')
    result = pytester.runpytest(str(p) + '::test_hello', '-h')
    result.stdout.fnmatch_lines('\n        *--hello-world*\n    ')

class TestConftestVisibility:

    def _setup_tree(self, pytester: Pytester) -> Dict[str, Path]:
        if False:
            for i in range(10):
                print('nop')
        runner = pytester.mkdir('empty')
        package = pytester.mkdir('package')
        package.joinpath('conftest.py').write_text(textwrap.dedent('                import pytest\n                @pytest.fixture\n                def fxtr():\n                    return "from-package"\n                '), encoding='utf-8')
        package.joinpath('test_pkgroot.py').write_text(textwrap.dedent('                def test_pkgroot(fxtr):\n                    assert fxtr == "from-package"\n                '), encoding='utf-8')
        swc = package.joinpath('swc')
        swc.mkdir()
        swc.joinpath('__init__.py').touch()
        swc.joinpath('conftest.py').write_text(textwrap.dedent('                import pytest\n                @pytest.fixture\n                def fxtr():\n                    return "from-swc"\n                '), encoding='utf-8')
        swc.joinpath('test_with_conftest.py').write_text(textwrap.dedent('                def test_with_conftest(fxtr):\n                    assert fxtr == "from-swc"\n                '), encoding='utf-8')
        snc = package.joinpath('snc')
        snc.mkdir()
        snc.joinpath('__init__.py').touch()
        snc.joinpath('test_no_conftest.py').write_text(textwrap.dedent('                def test_no_conftest(fxtr):\n                    assert fxtr == "from-package"   # No local conftest.py, so should\n                                                    # use value from parent dir\'s\n                '), encoding='utf-8')
        print('created directory structure:')
        for x in pytester.path.glob('**/'):
            print('   ' + str(x.relative_to(pytester.path)))
        return {'runner': runner, 'package': package, 'swc': swc, 'snc': snc}

    @pytest.mark.parametrize('chdir,testarg,expect_ntests_passed', [('runner', '..', 3), ('package', '..', 3), ('swc', '../..', 3), ('snc', '../..', 3), ('runner', '../package', 3), ('package', '.', 3), ('swc', '..', 3), ('snc', '..', 3), ('runner', '../package/swc', 1), ('package', './swc', 1), ('swc', '.', 1), ('snc', '../swc', 1), ('runner', '../package/snc', 1), ('package', './snc', 1), ('swc', '../snc', 1), ('snc', '.', 1)])
    def test_parsefactories_relative_node_ids(self, pytester: Pytester, chdir: str, testarg: str, expect_ntests_passed: int) -> None:
        if False:
            i = 10
            return i + 15
        '#616'
        dirs = self._setup_tree(pytester)
        print('pytest run in cwd: %s' % dirs[chdir].relative_to(pytester.path))
        print('pytestarg        : %s' % testarg)
        print('expected pass    : %s' % expect_ntests_passed)
        os.chdir(dirs[chdir])
        reprec = pytester.inline_run(testarg, '-q', '--traceconfig', '--confcutdir', pytester.path)
        reprec.assertoutcome(passed=expect_ntests_passed)

@pytest.mark.parametrize('confcutdir,passed,error', [('.', 2, 0), ('src', 1, 1), (None, 1, 1)])
def test_search_conftest_up_to_inifile(pytester: Pytester, confcutdir: str, passed: int, error: int) -> None:
    if False:
        print('Hello World!')
    'Test that conftest files are detected only up to an ini file, unless\n    an explicit --confcutdir option is given.\n    '
    root = pytester.path
    src = root.joinpath('src')
    src.mkdir()
    src.joinpath('pytest.ini').write_text('[pytest]', encoding='utf-8')
    src.joinpath('conftest.py').write_text(textwrap.dedent('            import pytest\n            @pytest.fixture\n            def fix1(): pass\n            '), encoding='utf-8')
    src.joinpath('test_foo.py').write_text(textwrap.dedent('            def test_1(fix1):\n                pass\n            def test_2(out_of_reach):\n                pass\n            '), encoding='utf-8')
    root.joinpath('conftest.py').write_text(textwrap.dedent('            import pytest\n            @pytest.fixture\n            def out_of_reach(): pass\n            '), encoding='utf-8')
    args = [str(src)]
    if confcutdir:
        args = ['--confcutdir=%s' % root.joinpath(confcutdir)]
    result = pytester.runpytest(*args)
    match = ''
    if passed:
        match += '*%d passed*' % passed
    if error:
        match += '*%d error*' % error
    result.stdout.fnmatch_lines(match)

def test_issue1073_conftest_special_objects(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makeconftest("        class DontTouchMe(object):\n            def __getattr__(self, x):\n                raise Exception('cant touch me')\n\n        x = DontTouchMe()\n        ")
    pytester.makepyfile('        def test_some():\n            pass\n        ')
    res = pytester.runpytest()
    assert res.ret == 0

def test_conftest_exception_handling(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makeconftest('        raise ValueError()\n        ')
    pytester.makepyfile('        def test_some():\n            pass\n        ')
    res = pytester.runpytest()
    assert res.ret == 4
    assert 'raise ValueError()' in [line.strip() for line in res.errlines]

def test_hook_proxy(pytester: Pytester) -> None:
    if False:
        return 10
    "Session's gethookproxy() would cache conftests incorrectly (#2016).\n    It was decided to remove the cache altogether.\n    "
    pytester.makepyfile(**{'root/demo-0/test_foo1.py': 'def test1(): pass', 'root/demo-a/test_foo2.py': 'def test1(): pass', 'root/demo-a/conftest.py': '            def pytest_ignore_collect(collection_path, config):\n                return True\n            ', 'root/demo-b/test_foo3.py': 'def test1(): pass', 'root/demo-c/test_foo4.py': 'def test1(): pass'})
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*test_foo1.py*', '*test_foo3.py*', '*test_foo4.py*', '*3 passed*'])

def test_required_option_help(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makeconftest('assert 0')
    x = pytester.mkdir('x')
    x.joinpath('conftest.py').write_text(textwrap.dedent('            def pytest_addoption(parser):\n                parser.addoption("--xyz", action="store_true", required=True)\n            '), encoding='utf-8')
    result = pytester.runpytest('-h', x)
    result.stdout.no_fnmatch_line('*argument --xyz is required*')
    assert 'general:' in result.stdout.str()