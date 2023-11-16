import os
import sys
import textwrap
from typing import Any
from typing import Dict
import _pytest._code
import pytest
from _pytest.config import ExitCode
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.pytester import Pytester
from _pytest.python import Class
from _pytest.python import Function

class TestModule:

    def test_failing_import(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        modcol = pytester.getmodulecol('import alksdjalskdjalkjals')
        pytest.raises(Collector.CollectError, modcol.collect)

    def test_import_duplicate(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        a = pytester.mkdir('a')
        b = pytester.mkdir('b')
        p1 = a.joinpath('test_whatever.py')
        p1.touch()
        p2 = b.joinpath('test_whatever.py')
        p2.touch()
        sys.modules.pop(p1.stem, None)
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*import*mismatch*', '*imported*test_whatever*', '*%s*' % p1, '*not the same*', '*%s*' % p2, '*HINT*'])

    def test_import_prepend_append(self, pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
        if False:
            i = 10
            return i + 15
        root1 = pytester.mkdir('root1')
        root2 = pytester.mkdir('root2')
        root1.joinpath('x456.py').touch()
        root2.joinpath('x456.py').touch()
        p = root2.joinpath('test_x456.py')
        monkeypatch.syspath_prepend(str(root1))
        p.write_text(textwrap.dedent('                import x456\n                def test():\n                    assert x456.__file__.startswith({!r})\n                '.format(str(root2))), encoding='utf-8')
        with monkeypatch.context() as mp:
            mp.chdir(root2)
            reprec = pytester.inline_run('--import-mode=append')
            reprec.assertoutcome(passed=0, failed=1)
            reprec = pytester.inline_run()
            reprec.assertoutcome(passed=1)

    def test_syntax_error_in_module(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        modcol = pytester.getmodulecol('this is a syntax error')
        pytest.raises(modcol.CollectError, modcol.collect)
        pytest.raises(modcol.CollectError, modcol.collect)

    def test_module_considers_pluginmanager_at_import(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        modcol = pytester.getmodulecol("pytest_plugins='xasdlkj',")
        pytest.raises(ImportError, lambda : modcol.obj)

    def test_invalid_test_module_name(self, pytester: Pytester) -> None:
        if False:
            return 10
        a = pytester.mkdir('a')
        a.joinpath('test_one.part1.py').touch()
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['ImportError while importing test module*test_one.part1*', 'Hint: make sure your test modules/packages have valid Python names.'])

    @pytest.mark.parametrize('verbose', [0, 1, 2])
    def test_show_traceback_import_error(self, pytester: Pytester, verbose: int) -> None:
        if False:
            while True:
                i = 10
        'Import errors when collecting modules should display the traceback (#1976).\n\n        With low verbosity we omit pytest and internal modules, otherwise show all traceback entries.\n        '
        pytester.makepyfile(foo_traceback_import_error='\n               from bar_traceback_import_error import NOT_AVAILABLE\n           ', bar_traceback_import_error='')
        pytester.makepyfile('\n               import foo_traceback_import_error\n        ')
        args = ('-v',) * verbose
        result = pytester.runpytest(*args)
        result.stdout.fnmatch_lines(['ImportError while importing test module*', 'Traceback:', '*from bar_traceback_import_error import NOT_AVAILABLE', '*cannot import name *NOT_AVAILABLE*'])
        assert result.ret == 2
        stdout = result.stdout.str()
        if verbose == 2:
            assert '_pytest' in stdout
        else:
            assert '_pytest' not in stdout

    def test_show_traceback_import_error_unicode(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        'Check test modules collected which raise ImportError with unicode messages\n        are handled properly (#2336).\n        '
        pytester.makepyfile("raise ImportError('Something bad happened ☺')")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['ImportError while importing test module*', 'Traceback:', '*raise ImportError*Something bad happened*'])
        assert result.ret == 2

class TestClass:

    def test_class_with_init_warning(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            class TestClass1(object):\n                def __init__(self):\n                    pass\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*cannot collect test class 'TestClass1' because it has a __init__ constructor (from: test_class_with_init_warning.py)"])

    def test_class_with_new_warning(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            class TestClass1(object):\n                def __new__(self):\n                    pass\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*cannot collect test class 'TestClass1' because it has a __new__ constructor (from: test_class_with_new_warning.py)"])

    def test_class_subclassobject(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.getmodulecol('\n            class test(object):\n                pass\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*collected 0*'])

    def test_static_method(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        'Support for collecting staticmethod tests (#2528, #2699)'
        pytester.getmodulecol('\n            import pytest\n            class Test(object):\n                @staticmethod\n                def test_something():\n                    pass\n\n                @pytest.fixture\n                def fix(self):\n                    return 1\n\n                @staticmethod\n                def test_fix(fix):\n                    assert fix == 1\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*collected 2 items*', '*2 passed in*'])

    def test_setup_teardown_class_as_classmethod(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile(test_mod1='\n            class TestClassMethod(object):\n                @classmethod\n                def setup_class(cls):\n                    pass\n                def test_1(self):\n                    pass\n                @classmethod\n                def teardown_class(cls):\n                    pass\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*1 passed*'])

    def test_issue1035_obj_has_getattr(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        modcol = pytester.getmodulecol('\n            class Chameleon(object):\n                def __getattr__(self, name):\n                    return True\n            chameleon = Chameleon()\n        ')
        colitems = modcol.collect()
        assert len(colitems) == 0

    def test_issue1579_namedtuple(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile("\n            import collections\n\n            TestCase = collections.namedtuple('TestCase', ['a'])\n        ")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines("*cannot collect test class 'TestCase' because it has a __new__ constructor*")

    def test_issue2234_property(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            class TestCase(object):\n                @property\n                def prop(self):\n                    raise NotImplementedError()\n        ')
        result = pytester.runpytest()
        assert result.ret == ExitCode.NO_TESTS_COLLECTED

class TestFunction:

    def test_getmodulecollector(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        item = pytester.getitem('def test_func(): pass')
        modcol = item.getparent(pytest.Module)
        assert isinstance(modcol, pytest.Module)
        assert hasattr(modcol.obj, 'test_func')

    @pytest.mark.filterwarnings('default')
    def test_function_as_object_instance_ignored(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            class A(object):\n                def __call__(self, tmp_path):\n                    0/0\n\n            test_a = A()\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['collected 0 items', "*test_function_as_object_instance_ignored.py:2: *cannot collect 'test_a' because it is not a function."])

    @staticmethod
    def make_function(pytester: Pytester, **kwargs: Any) -> Any:
        if False:
            i = 10
            return i + 15
        from _pytest.fixtures import FixtureManager
        config = pytester.parseconfigure()
        session = Session.from_config(config)
        session._fixturemanager = FixtureManager(session)
        return pytest.Function.from_parent(parent=session, **kwargs)

    def test_function_equality(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10

        def func1():
            if False:
                for i in range(10):
                    print('nop')
            pass

        def func2():
            if False:
                print('Hello World!')
            pass
        f1 = self.make_function(pytester, name='name', callobj=func1)
        assert f1 == f1
        f2 = self.make_function(pytester, name='name', callobj=func2, originalname='foobar')
        assert f1 != f2

    def test_repr_produces_actual_test_id(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        f = self.make_function(pytester, name='test[\\xe5]', callobj=self.test_repr_produces_actual_test_id)
        assert repr(f) == '<Function test[\\xe5]>'

    def test_issue197_parametrize_emptyset(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile("\n            import pytest\n            @pytest.mark.parametrize('arg', [])\n            def test_function(arg):\n                pass\n        ")
        reprec = pytester.inline_run()
        reprec.assertoutcome(skipped=1)

    def test_single_tuple_unwraps_values(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile("\n            import pytest\n            @pytest.mark.parametrize(('arg',), [(1,)])\n            def test_function(arg):\n                assert arg == 1\n        ")
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)

    def test_issue213_parametrize_value_no_equal(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            import pytest\n            class A(object):\n                def __eq__(self, other):\n                    raise ValueError("not possible")\n            @pytest.mark.parametrize(\'arg\', [A()])\n            def test_function(arg):\n                assert arg.__class__.__name__ == "A"\n        ')
        reprec = pytester.inline_run('--fulltrace')
        reprec.assertoutcome(passed=1)

    def test_parametrize_with_non_hashable_values(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        'Test parametrization with non-hashable values.'
        pytester.makepyfile("\n            archival_mapping = {\n                '1.0': {'tag': '1.0'},\n                '1.2.2a1': {'tag': 'release-1.2.2a1'},\n            }\n\n            import pytest\n            @pytest.mark.parametrize('key value'.split(),\n                                     archival_mapping.items())\n            def test_archival_to_version(key, value):\n                assert key in archival_mapping\n                assert value == archival_mapping[key]\n        ")
        rec = pytester.inline_run()
        rec.assertoutcome(passed=2)

    def test_parametrize_with_non_hashable_values_indirect(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        'Test parametrization with non-hashable values with indirect parametrization.'
        pytester.makepyfile("\n            archival_mapping = {\n                '1.0': {'tag': '1.0'},\n                '1.2.2a1': {'tag': 'release-1.2.2a1'},\n            }\n\n            import pytest\n\n            @pytest.fixture\n            def key(request):\n                return request.param\n\n            @pytest.fixture\n            def value(request):\n                return request.param\n\n            @pytest.mark.parametrize('key value'.split(),\n                                     archival_mapping.items(), indirect=True)\n            def test_archival_to_version(key, value):\n                assert key in archival_mapping\n                assert value == archival_mapping[key]\n        ")
        rec = pytester.inline_run()
        rec.assertoutcome(passed=2)

    def test_parametrize_overrides_fixture(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test parametrization when parameter overrides existing fixture with same name.'
        pytester.makepyfile("\n            import pytest\n\n            @pytest.fixture\n            def value():\n                return 'value'\n\n            @pytest.mark.parametrize('value',\n                                     ['overridden'])\n            def test_overridden_via_param(value):\n                assert value == 'overridden'\n\n            @pytest.mark.parametrize('somevalue', ['overridden'])\n            def test_not_overridden(value, somevalue):\n                assert value == 'value'\n                assert somevalue == 'overridden'\n\n            @pytest.mark.parametrize('other,value', [('foo', 'overridden')])\n            def test_overridden_via_multiparam(other, value):\n                assert other == 'foo'\n                assert value == 'overridden'\n        ")
        rec = pytester.inline_run()
        rec.assertoutcome(passed=3)

    def test_parametrize_overrides_parametrized_fixture(self, pytester: Pytester) -> None:
        if False:
            return 10
        'Test parametrization when parameter overrides existing parametrized fixture with same name.'
        pytester.makepyfile("\n            import pytest\n\n            @pytest.fixture(params=[1, 2])\n            def value(request):\n                return request.param\n\n            @pytest.mark.parametrize('value',\n                                     ['overridden'])\n            def test_overridden_via_param(value):\n                assert value == 'overridden'\n        ")
        rec = pytester.inline_run()
        rec.assertoutcome(passed=1)

    def test_parametrize_overrides_indirect_dependency_fixture(self, pytester: Pytester) -> None:
        if False:
            return 10
        'Test parametrization when parameter overrides a fixture that a test indirectly depends on'
        pytester.makepyfile("\n            import pytest\n\n            fix3_instantiated = False\n\n            @pytest.fixture\n            def fix1(fix2):\n               return fix2 + '1'\n\n            @pytest.fixture\n            def fix2(fix3):\n               return fix3 + '2'\n\n            @pytest.fixture\n            def fix3():\n               global fix3_instantiated\n               fix3_instantiated = True\n               return '3'\n\n            @pytest.mark.parametrize('fix2', ['2'])\n            def test_it(fix1):\n               assert fix1 == '21'\n               assert not fix3_instantiated\n        ")
        rec = pytester.inline_run()
        rec.assertoutcome(passed=1)

    def test_parametrize_with_mark(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        items = pytester.getitems("\n            import pytest\n            @pytest.mark.foo\n            @pytest.mark.parametrize('arg', [\n                1,\n                pytest.param(2, marks=[pytest.mark.baz, pytest.mark.bar])\n            ])\n            def test_function(arg):\n                pass\n        ")
        keywords = [item.keywords for item in items]
        assert 'foo' in keywords[0] and 'bar' not in keywords[0] and ('baz' not in keywords[0])
        assert 'foo' in keywords[1] and 'bar' in keywords[1] and ('baz' in keywords[1])

    def test_parametrize_with_empty_string_arguments(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        items = pytester.getitems("            import pytest\n\n            @pytest.mark.parametrize('v', ('', ' '))\n            @pytest.mark.parametrize('w', ('', ' '))\n            def test(v, w): ...\n            ")
        names = {item.name for item in items}
        assert names == {'test[-]', 'test[ -]', 'test[- ]', 'test[ - ]'}

    def test_function_equality_with_callspec(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        items = pytester.getitems("\n            import pytest\n            @pytest.mark.parametrize('arg', [1,2])\n            def test_function(arg):\n                pass\n        ")
        assert items[0] != items[1]
        assert not items[0] == items[1]

    def test_pyfunc_call(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        item = pytester.getitem('def test_func(): raise ValueError')
        config = item.config

        class MyPlugin1:

            def pytest_pyfunc_call(self):
                if False:
                    return 10
                raise ValueError

        class MyPlugin2:

            def pytest_pyfunc_call(self):
                if False:
                    print('Hello World!')
                return True
        config.pluginmanager.register(MyPlugin1())
        config.pluginmanager.register(MyPlugin2())
        config.hook.pytest_runtest_setup(item=item)
        config.hook.pytest_pyfunc_call(pyfuncitem=item)

    def test_multiple_parametrize(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        modcol = pytester.getmodulecol("\n            import pytest\n            @pytest.mark.parametrize('x', [0, 1])\n            @pytest.mark.parametrize('y', [2, 3])\n            def test1(x, y):\n                pass\n        ")
        colitems = modcol.collect()
        assert colitems[0].name == 'test1[2-0]'
        assert colitems[1].name == 'test1[2-1]'
        assert colitems[2].name == 'test1[3-0]'
        assert colitems[3].name == 'test1[3-1]'

    def test_issue751_multiple_parametrize_with_ids(self, pytester: Pytester) -> None:
        if False:
            return 10
        modcol = pytester.getmodulecol("\n            import pytest\n            @pytest.mark.parametrize('x', [0], ids=['c'])\n            @pytest.mark.parametrize('y', [0, 1], ids=['a', 'b'])\n            class Test(object):\n                def test1(self, x, y):\n                    pass\n                def test2(self, x, y):\n                    pass\n        ")
        colitems = modcol.collect()[0].collect()
        assert colitems[0].name == 'test1[a-c]'
        assert colitems[1].name == 'test1[b-c]'
        assert colitems[2].name == 'test2[a-c]'
        assert colitems[3].name == 'test2[b-c]'

    def test_parametrize_skipif(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile("\n            import pytest\n\n            m = pytest.mark.skipif('True')\n\n            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])\n            def test_skip_if(x):\n                assert x < 2\n        ")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 2 passed, 1 skipped in *'])

    def test_parametrize_skip(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile("\n            import pytest\n\n            m = pytest.mark.skip('')\n\n            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])\n            def test_skip(x):\n                assert x < 2\n        ")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 2 passed, 1 skipped in *'])

    def test_parametrize_skipif_no_skip(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile("\n            import pytest\n\n            m = pytest.mark.skipif('False')\n\n            @pytest.mark.parametrize('x', [0, 1, m(2)])\n            def test_skipif_no_skip(x):\n                assert x < 2\n        ")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 1 failed, 2 passed in *'])

    def test_parametrize_xfail(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile("\n            import pytest\n\n            m = pytest.mark.xfail('True')\n\n            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])\n            def test_xfail(x):\n                assert x < 2\n        ")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 2 passed, 1 xfailed in *'])

    def test_parametrize_passed(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile("\n            import pytest\n\n            m = pytest.mark.xfail('True')\n\n            @pytest.mark.parametrize('x', [0, 1, pytest.param(2, marks=m)])\n            def test_xfail(x):\n                pass\n        ")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 2 passed, 1 xpassed in *'])

    def test_parametrize_xfail_passed(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile("\n            import pytest\n\n            m = pytest.mark.xfail('False')\n\n            @pytest.mark.parametrize('x', [0, 1, m(2)])\n            def test_passed(x):\n                pass\n        ")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 3 passed in *'])

    def test_function_originalname(self, pytester: Pytester) -> None:
        if False:
            return 10
        items = pytester.getitems("\n            import pytest\n\n            @pytest.mark.parametrize('arg', [1,2])\n            def test_func(arg):\n                pass\n\n            def test_no_param():\n                pass\n        ")
        originalnames = []
        for x in items:
            assert isinstance(x, pytest.Function)
            originalnames.append(x.originalname)
        assert originalnames == ['test_func', 'test_func', 'test_no_param']

    def test_function_with_square_brackets(self, pytester: Pytester) -> None:
        if False:
            return 10
        "Check that functions with square brackets don't cause trouble."
        p1 = pytester.makepyfile('\n            locals()["test_foo[name]"] = lambda: None\n            ')
        result = pytester.runpytest('-v', str(p1))
        result.stdout.fnmatch_lines(['test_function_with_square_brackets.py::test_foo[[]name[]] PASSED *', '*= 1 passed in *'])

class TestSorting:

    def test_check_equality(self, pytester: Pytester) -> None:
        if False:
            return 10
        modcol = pytester.getmodulecol('\n            def test_pass(): pass\n            def test_fail(): assert 0\n        ')
        fn1 = pytester.collect_by_name(modcol, 'test_pass')
        assert isinstance(fn1, pytest.Function)
        fn2 = pytester.collect_by_name(modcol, 'test_pass')
        assert isinstance(fn2, pytest.Function)
        assert fn1 == fn2
        assert fn1 != modcol
        assert hash(fn1) == hash(fn2)
        fn3 = pytester.collect_by_name(modcol, 'test_fail')
        assert isinstance(fn3, pytest.Function)
        assert not fn1 == fn3
        assert fn1 != fn3
        for fn in (fn1, fn2, fn3):
            assert fn != 3
            assert fn != modcol
            assert fn != [1, 2, 3]
            assert [1, 2, 3] != fn
            assert modcol != fn

    def test_allow_sane_sorting_for_decorators(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        modcol = pytester.getmodulecol('\n            def dec(f):\n                g = lambda: f(2)\n                g.place_as = f\n                return g\n\n\n            def test_b(y):\n                pass\n            test_b = dec(test_b)\n\n            def test_a(y):\n                pass\n            test_a = dec(test_a)\n        ')
        colitems = modcol.collect()
        assert len(colitems) == 2
        assert [item.name for item in colitems] == ['test_b', 'test_a']

    def test_ordered_by_definition_order(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('            class Test1:\n                def test_foo(self): pass\n                def test_bar(self): pass\n            class Test2:\n                def test_foo(self): pass\n                test_bar = Test1.test_bar\n            class Test3(Test2):\n                def test_baz(self): pass\n            ')
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(['*Class Test1*', '*Function test_foo*', '*Function test_bar*', '*Class Test2*', '*Function test_foo*', '*Function test_bar*', '*Class Test3*', '*Function test_foo*', '*Function test_bar*', '*Function test_baz*'])

class TestConftestCustomization:

    def test_pytest_pycollect_module(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makeconftest('\n            import pytest\n            class MyModule(pytest.Module):\n                pass\n            def pytest_pycollect_makemodule(module_path, parent):\n                if module_path.name == "test_xyz.py":\n                    return MyModule.from_parent(path=module_path, parent=parent)\n        ')
        pytester.makepyfile('def test_some(): pass')
        pytester.makepyfile(test_xyz='def test_func(): pass')
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(['*<Module*test_pytest*', '*<MyModule*xyz*'])

    def test_customized_pymakemodule_issue205_subdir(self, pytester: Pytester) -> None:
        if False:
            return 10
        b = pytester.path.joinpath('a', 'b')
        b.mkdir(parents=True)
        b.joinpath('conftest.py').write_text(textwrap.dedent('                import pytest\n                @pytest.hookimpl(wrapper=True)\n                def pytest_pycollect_makemodule():\n                    mod = yield\n                    mod.obj.hello = "world"\n                    return mod\n                '), encoding='utf-8')
        b.joinpath('test_module.py').write_text(textwrap.dedent('                def test_hello():\n                    assert hello == "world"\n                '), encoding='utf-8')
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)

    def test_customized_pymakeitem(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        b = pytester.path.joinpath('a', 'b')
        b.mkdir(parents=True)
        b.joinpath('conftest.py').write_text(textwrap.dedent('                import pytest\n                @pytest.hookimpl(wrapper=True)\n                def pytest_pycollect_makeitem():\n                    result = yield\n                    if result:\n                        for func in result:\n                            func._some123 = "world"\n                    return result\n                '), encoding='utf-8')
        b.joinpath('test_module.py').write_text(textwrap.dedent('                import pytest\n\n                @pytest.fixture()\n                def obj(request):\n                    return request.node._some123\n                def test_hello(obj):\n                    assert obj == "world"\n                '), encoding='utf-8')
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)

    def test_pytest_pycollect_makeitem(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makeconftest('\n            import pytest\n            class MyFunction(pytest.Function):\n                pass\n            def pytest_pycollect_makeitem(collector, name, obj):\n                if name == "some":\n                    return MyFunction.from_parent(name=name, parent=collector)\n        ')
        pytester.makepyfile('def some(): pass')
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(['*MyFunction*some*'])

    def test_issue2369_collect_module_fileext(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Ensure we can collect files with weird file extensions as Python\n        modules (#2369)'
        pytester.makeconftest('\n            import sys\n            import os.path\n            from importlib.util import spec_from_loader\n            from importlib.machinery import SourceFileLoader\n            from _pytest.python import Module\n\n            class MetaPathFinder:\n                def find_spec(self, fullname, path, target=None):\n                    if os.path.exists(fullname + ".narf"):\n                        return spec_from_loader(\n                            fullname,\n                            SourceFileLoader(fullname, fullname + ".narf"),\n                        )\n            sys.meta_path.append(MetaPathFinder())\n\n            def pytest_collect_file(file_path, parent):\n                if file_path.suffix == ".narf":\n                    return Module.from_parent(path=file_path, parent=parent)\n            ')
        pytester.makefile('.narf', '            def test_something():\n                assert 1 + 1 == 2')
        result = pytester.runpytest_subprocess()
        result.stdout.fnmatch_lines(['*1 passed*'])

    def test_early_ignored_attributes(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        'Builtin attributes should be ignored early on, even if\n        configuration would otherwise allow them.\n\n        This tests a performance optimization, not correctness, really,\n        although it tests PytestCollectionWarning is not raised, while\n        it would have been raised otherwise.\n        '
        pytester.makeini('\n            [pytest]\n            python_classes=*\n            python_functions=*\n        ')
        pytester.makepyfile('\n            class TestEmpty:\n                pass\n            test_empty = TestEmpty()\n            def test_real():\n                pass\n        ')
        (items, rec) = pytester.inline_genitems()
        assert rec.ret == 0
        assert len(items) == 1

def test_setup_only_available_in_subdir(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    sub1 = pytester.mkpydir('sub1')
    sub2 = pytester.mkpydir('sub2')
    sub1.joinpath('conftest.py').write_text(textwrap.dedent('            import pytest\n            def pytest_runtest_setup(item):\n                assert item.path.stem == "test_in_sub1"\n            def pytest_runtest_call(item):\n                assert item.path.stem == "test_in_sub1"\n            def pytest_runtest_teardown(item):\n                assert item.path.stem == "test_in_sub1"\n            '), encoding='utf-8')
    sub2.joinpath('conftest.py').write_text(textwrap.dedent('            import pytest\n            def pytest_runtest_setup(item):\n                assert item.path.stem == "test_in_sub2"\n            def pytest_runtest_call(item):\n                assert item.path.stem == "test_in_sub2"\n            def pytest_runtest_teardown(item):\n                assert item.path.stem == "test_in_sub2"\n            '), encoding='utf-8')
    sub1.joinpath('test_in_sub1.py').write_text('def test_1(): pass', encoding='utf-8')
    sub2.joinpath('test_in_sub2.py').write_text('def test_2(): pass', encoding='utf-8')
    result = pytester.runpytest('-v', '-s')
    result.assert_outcomes(passed=2)

def test_modulecol_roundtrip(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    modcol = pytester.getmodulecol('pass', withinit=False)
    trail = modcol.nodeid
    newcol = modcol.session.perform_collect([trail], genitems=0)[0]
    assert modcol.name == newcol.name

class TestTracebackCutting:

    def test_skip_simple(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(pytest.skip.Exception) as excinfo:
            pytest.skip('xxx')
        assert excinfo.traceback[-1].frame.code.name == 'skip'
        assert excinfo.traceback[-1].ishidden(excinfo)
        assert excinfo.traceback[-2].frame.code.name == 'test_skip_simple'
        assert not excinfo.traceback[-2].ishidden(excinfo)

    def test_traceback_argsetup(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makeconftest('\n            import pytest\n\n            @pytest.fixture\n            def hello(request):\n                raise ValueError("xyz")\n        ')
        p = pytester.makepyfile('def test(hello): pass')
        result = pytester.runpytest(p)
        assert result.ret != 0
        out = result.stdout.str()
        assert 'xyz' in out
        assert 'conftest.py:5: ValueError' in out
        numentries = out.count('_ _ _')
        assert numentries == 0
        result = pytester.runpytest('--fulltrace', p)
        out = result.stdout.str()
        assert 'conftest.py:5: ValueError' in out
        numentries = out.count('_ _ _ _')
        assert numentries > 3

    def test_traceback_error_during_import(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            x = 1\n            x = 2\n            x = 17\n            asd\n        ')
        result = pytester.runpytest()
        assert result.ret != 0
        out = result.stdout.str()
        assert 'x = 1' not in out
        assert 'x = 2' not in out
        result.stdout.fnmatch_lines([' *asd*', 'E*NameError*'])
        result = pytester.runpytest('--fulltrace')
        out = result.stdout.str()
        assert 'x = 1' in out
        assert 'x = 2' in out
        result.stdout.fnmatch_lines(['>*asd*', 'E*NameError*'])

    def test_traceback_filter_error_during_fixture_collection(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        'Integration test for issue #995.'
        pytester.makepyfile('\n            import pytest\n\n            def fail_me(func):\n                ns = {}\n                exec(\'def w(): raise ValueError("fail me")\', ns)\n                return ns[\'w\']\n\n            @pytest.fixture(scope=\'class\')\n            @fail_me\n            def fail_fixture():\n                pass\n\n            def test_failing_fixture(fail_fixture):\n               pass\n        ')
        result = pytester.runpytest()
        assert result.ret != 0
        out = result.stdout.str()
        assert 'INTERNALERROR>' not in out
        result.stdout.fnmatch_lines(['*ValueError: fail me*', '* 1 error in *'])

    def test_filter_traceback_generated_code(self) -> None:
        if False:
            return 10
        'Test that filter_traceback() works with the fact that\n        _pytest._code.code.Code.path attribute might return an str object.\n\n        In this case, one of the entries on the traceback was produced by\n        dynamically generated code.\n        See: https://bitbucket.org/pytest-dev/py/issues/71\n        This fixes #995.\n        '
        from _pytest._code import filter_traceback
        tb = None
        try:
            ns: Dict[str, Any] = {}
            exec('def foo(): raise ValueError', ns)
            ns['foo']()
        except ValueError:
            (_, _, tb) = sys.exc_info()
        assert tb is not None
        traceback = _pytest._code.Traceback(tb)
        assert isinstance(traceback[-1].path, str)
        assert not filter_traceback(traceback[-1])

    def test_filter_traceback_path_no_longer_valid(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        'Test that filter_traceback() works with the fact that\n        _pytest._code.code.Code.path attribute might return an str object.\n\n        In this case, one of the files in the traceback no longer exists.\n        This fixes #1133.\n        '
        from _pytest._code import filter_traceback
        pytester.syspathinsert()
        pytester.makepyfile(filter_traceback_entry_as_str='\n            def foo():\n                raise ValueError\n        ')
        tb = None
        try:
            import filter_traceback_entry_as_str
            filter_traceback_entry_as_str.foo()
        except ValueError:
            (_, _, tb) = sys.exc_info()
        assert tb is not None
        pytester.path.joinpath('filter_traceback_entry_as_str.py').unlink()
        traceback = _pytest._code.Traceback(tb)
        assert isinstance(traceback[-1].path, str)
        assert filter_traceback(traceback[-1])

class TestReportInfo:

    def test_itemreport_reportinfo(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makeconftest('\n            import pytest\n            class MyFunction(pytest.Function):\n                def reportinfo(self):\n                    return "ABCDE", 42, "custom"\n            def pytest_pycollect_makeitem(collector, name, obj):\n                if name == "test_func":\n                    return MyFunction.from_parent(name=name, parent=collector)\n        ')
        item = pytester.getitem('def test_func(): pass')
        item.config.pluginmanager.getplugin('runner')
        assert item.location == ('ABCDE', 42, 'custom')

    def test_func_reportinfo(self, pytester: Pytester) -> None:
        if False:
            return 10
        item = pytester.getitem('def test_func(): pass')
        (path, lineno, modpath) = item.reportinfo()
        assert os.fspath(path) == str(item.path)
        assert lineno == 0
        assert modpath == 'test_func'

    def test_class_reportinfo(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        modcol = pytester.getmodulecol('\n            # lineno 0\n            class TestClass(object):\n                def test_hello(self): pass\n        ')
        classcol = pytester.collect_by_name(modcol, 'TestClass')
        assert isinstance(classcol, Class)
        (path, lineno, msg) = classcol.reportinfo()
        assert os.fspath(path) == str(modcol.path)
        assert lineno == 1
        assert msg == 'TestClass'

    @pytest.mark.filterwarnings('ignore:usage of Generator.Function is deprecated, please use pytest.Function instead')
    def test_reportinfo_with_nasty_getattr(self, pytester: Pytester) -> None:
        if False:
            return 10
        modcol = pytester.getmodulecol('\n            # lineno 0\n            class TestClass:\n                def __getattr__(self, name):\n                    return "this is not an int"\n\n                def __class_getattr__(cls, name):\n                    return "this is not an int"\n\n                def intest_foo(self):\n                    pass\n\n                def test_bar(self):\n                    pass\n        ')
        classcol = pytester.collect_by_name(modcol, 'TestClass')
        assert isinstance(classcol, Class)
        (path, lineno, msg) = classcol.reportinfo()
        func = list(classcol.collect())[0]
        assert isinstance(func, Function)
        (path, lineno, msg) = func.reportinfo()

def test_customized_python_discovery(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makeini('\n        [pytest]\n        python_files=check_*.py\n        python_classes=Check\n        python_functions=check\n    ')
    p = pytester.makepyfile('\n        def check_simple():\n            pass\n        class CheckMyApp(object):\n            def check_meth(self):\n                pass\n    ')
    p2 = p.with_name(p.name.replace('test', 'check'))
    p.rename(p2)
    result = pytester.runpytest('--collect-only', '-s')
    result.stdout.fnmatch_lines(['*check_customized*', '*check_simple*', '*CheckMyApp*', '*check_meth*'])
    result = pytester.runpytest()
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*2 passed*'])

def test_customized_python_discovery_functions(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makeini('\n        [pytest]\n        python_functions=_test\n    ')
    pytester.makepyfile('\n        def _test_underscore():\n            pass\n    ')
    result = pytester.runpytest('--collect-only', '-s')
    result.stdout.fnmatch_lines(['*_test_underscore*'])
    result = pytester.runpytest()
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*1 passed*'])

def test_unorderable_types(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        class TestJoinEmpty(object):\n            pass\n\n        def make_test():\n            class Test(object):\n                pass\n            Test.__name__ = "TestFoo"\n            return Test\n        TestFoo = make_test()\n    ')
    result = pytester.runpytest()
    result.stdout.no_fnmatch_line('*TypeError*')
    assert result.ret == ExitCode.NO_TESTS_COLLECTED

@pytest.mark.filterwarnings('default::pytest.PytestCollectionWarning')
def test_dont_collect_non_function_callable(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Test for issue https://github.com/pytest-dev/pytest/issues/331\n\n    In this case an INTERNALERROR occurred trying to report the failure of\n    a test like this one because pytest failed to get the source lines.\n    '
    pytester.makepyfile('\n        class Oh(object):\n            def __call__(self):\n                pass\n\n        test_a = Oh()\n\n        def test_real():\n            pass\n    ')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*collected 1 item*', "*test_dont_collect_non_function_callable.py:2: *cannot collect 'test_a' because it is not a function*", '*1 passed, 1 warning in *'])

def test_class_injection_does_not_break_collection(pytester: Pytester) -> None:
    if False:
        return 10
    'Tests whether injection during collection time will terminate testing.\n\n    In this case the error should not occur if the TestClass itself\n    is modified during collection time, and the original method list\n    is still used for collection.\n    '
    pytester.makeconftest('\n        from test_inject import TestClass\n        def pytest_generate_tests(metafunc):\n            TestClass.changed_var = {}\n    ')
    pytester.makepyfile(test_inject='\n         class TestClass(object):\n            def test_injection(self):\n                """Test being parametrized."""\n                pass\n    ')
    result = pytester.runpytest()
    assert 'RuntimeError: dictionary changed size during iteration' not in result.stdout.str()
    result.stdout.fnmatch_lines(['*1 passed*'])

def test_syntax_error_with_non_ascii_chars(pytester: Pytester) -> None:
    if False:
        return 10
    'Fix decoding issue while formatting SyntaxErrors during collection (#578).'
    pytester.makepyfile('☃')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*ERROR collecting*', '*SyntaxError*', '*1 error in*'])

def test_collect_error_with_fulltrace(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('assert 0')
    result = pytester.runpytest('--fulltrace')
    result.stdout.fnmatch_lines(['collected 0 items / 1 error', '', '*= ERRORS =*', '*_ ERROR collecting test_collect_error_with_fulltrace.py _*', '', '>   assert 0', 'E   assert 0', '', 'test_collect_error_with_fulltrace.py:1: AssertionError', '*! Interrupted: 1 error during collection !*'])

def test_skip_duplicates_by_default(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Test for issue https://github.com/pytest-dev/pytest/issues/1609 (#1609)\n\n    Ignore duplicate directories.\n    '
    a = pytester.mkdir('a')
    fh = a.joinpath('test_a.py')
    fh.write_text(textwrap.dedent('            import pytest\n            def test_real():\n                pass\n            '), encoding='utf-8')
    result = pytester.runpytest(str(a), str(a))
    result.stdout.fnmatch_lines(['*collected 1 item*'])

def test_keep_duplicates(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    'Test for issue https://github.com/pytest-dev/pytest/issues/1609 (#1609)\n\n    Use --keep-duplicates to collect tests from duplicate directories.\n    '
    a = pytester.mkdir('a')
    fh = a.joinpath('test_a.py')
    fh.write_text(textwrap.dedent('            import pytest\n            def test_real():\n                pass\n            '), encoding='utf-8')
    result = pytester.runpytest('--keep-duplicates', str(a), str(a))
    result.stdout.fnmatch_lines(['*collected 2 item*'])

def test_package_collection_infinite_recursion(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.copy_example('collect/package_infinite_recursion')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['*1 passed*'])

def test_package_collection_init_given_as_argument(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Regression test for #3749, #8976, #9263, #9313.\n\n    Specifying an __init__.py file directly should collect only the __init__.py\n    Module, not the entire package.\n    '
    p = pytester.copy_example('collect/package_init_given_as_arg')
    (items, hookrecorder) = pytester.inline_genitems(p / 'pkg' / '__init__.py')
    assert len(items) == 1
    assert items[0].name == 'test_init'

def test_package_with_modules(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    '\n    .\n    └── root\n        ├── __init__.py\n        ├── sub1\n        │   ├── __init__.py\n        │   └── sub1_1\n        │       ├── __init__.py\n        │       └── test_in_sub1.py\n        └── sub2\n            └── test\n                └── test_in_sub2.py\n\n    '
    root = pytester.mkpydir('root')
    sub1 = root.joinpath('sub1')
    sub1_test = sub1.joinpath('sub1_1')
    sub1_test.mkdir(parents=True)
    for d in (sub1, sub1_test):
        d.joinpath('__init__.py').touch()
    sub2 = root.joinpath('sub2')
    sub2_test = sub2.joinpath('test')
    sub2_test.mkdir(parents=True)
    sub1_test.joinpath('test_in_sub1.py').write_text('def test_1(): pass', encoding='utf-8')
    sub2_test.joinpath('test_in_sub2.py').write_text('def test_2(): pass', encoding='utf-8')
    result = pytester.runpytest('-v', '-s')
    result.assert_outcomes(passed=2)
    result = pytester.runpytest('-v', '-s', 'root')
    result.assert_outcomes(passed=2)
    os.chdir(root)
    result = pytester.runpytest('-v', '-s')
    result.assert_outcomes(passed=2)

def test_package_ordering(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    '\n    .\n    └── root\n        ├── Test_root.py\n        ├── __init__.py\n        ├── sub1\n        │   ├── Test_sub1.py\n        │   └── __init__.py\n        └── sub2\n            └── test\n                └── test_sub2.py\n\n    '
    pytester.makeini('\n        [pytest]\n        python_files=*.py\n    ')
    root = pytester.mkpydir('root')
    sub1 = root.joinpath('sub1')
    sub1.mkdir()
    sub1.joinpath('__init__.py').touch()
    sub2 = root.joinpath('sub2')
    sub2_test = sub2.joinpath('test')
    sub2_test.mkdir(parents=True)
    root.joinpath('Test_root.py').write_text('def test_1(): pass', encoding='utf-8')
    sub1.joinpath('Test_sub1.py').write_text('def test_2(): pass', encoding='utf-8')
    sub2_test.joinpath('test_sub2.py').write_text('def test_3(): pass', encoding='utf-8')
    result = pytester.runpytest('-v', '-s')
    result.assert_outcomes(passed=3)