import pytest
from _pytest.pytester import Pytester

def setup_module(mod):
    if False:
        i = 10
        return i + 15
    mod.nose = pytest.importorskip('nose')

def test_nose_setup(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    p = pytester.makepyfile('\n        values = []\n        from nose.tools import with_setup\n\n        @with_setup(lambda: values.append(1), lambda: values.append(2))\n        def test_hello():\n            assert values == [1]\n\n        def test_world():\n            assert values == [1,2]\n\n        test_hello.setup = lambda: values.append(1)\n        test_hello.teardown = lambda: values.append(2)\n    ')
    result = pytester.runpytest(p, '-p', 'nose', '-Wignore::pytest.PytestRemovedIn8Warning')
    result.assert_outcomes(passed=2)

def test_setup_func_with_setup_decorator() -> None:
    if False:
        return 10
    from _pytest.nose import call_optional
    values = []

    class A:

        @pytest.fixture(autouse=True)
        def f(self):
            if False:
                i = 10
                return i + 15
            values.append(1)
    call_optional(A(), 'f', 'A.f')
    assert not values

def test_setup_func_not_callable() -> None:
    if False:
        i = 10
        return i + 15
    from _pytest.nose import call_optional

    class A:
        f = 1
    call_optional(A(), 'f', 'A.f')

def test_nose_setup_func(pytester: Pytester) -> None:
    if False:
        return 10
    p = pytester.makepyfile('\n        from nose.tools import with_setup\n\n        values = []\n\n        def my_setup():\n            a = 1\n            values.append(a)\n\n        def my_teardown():\n            b = 2\n            values.append(b)\n\n        @with_setup(my_setup, my_teardown)\n        def test_hello():\n            print(values)\n            assert values == [1]\n\n        def test_world():\n            print(values)\n            assert values == [1,2]\n\n    ')
    result = pytester.runpytest(p, '-p', 'nose', '-Wignore::pytest.PytestRemovedIn8Warning')
    result.assert_outcomes(passed=2)

def test_nose_setup_func_failure(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    p = pytester.makepyfile('\n        from nose.tools import with_setup\n\n        values = []\n        my_setup = lambda x: 1\n        my_teardown = lambda x: 2\n\n        @with_setup(my_setup, my_teardown)\n        def test_hello():\n            print(values)\n            assert values == [1]\n\n        def test_world():\n            print(values)\n            assert values == [1,2]\n\n    ')
    result = pytester.runpytest(p, '-p', 'nose', '-Wignore::pytest.PytestRemovedIn8Warning')
    result.stdout.fnmatch_lines(['*TypeError: <lambda>()*'])

def test_nose_setup_func_failure_2(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile('\n        values = []\n\n        my_setup = 1\n        my_teardown = 2\n\n        def test_hello():\n            assert values == []\n\n        test_hello.setup = my_setup\n        test_hello.teardown = my_teardown\n    ')
    reprec = pytester.inline_run()
    reprec.assertoutcome(passed=1)

def test_nose_setup_partial(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    pytest.importorskip('functools')
    p = pytester.makepyfile('\n        from functools import partial\n\n        values = []\n\n        def my_setup(x):\n            a = x\n            values.append(a)\n\n        def my_teardown(x):\n            b = x\n            values.append(b)\n\n        my_setup_partial = partial(my_setup, 1)\n        my_teardown_partial = partial(my_teardown, 2)\n\n        def test_hello():\n            print(values)\n            assert values == [1]\n\n        def test_world():\n            print(values)\n            assert values == [1,2]\n\n        test_hello.setup = my_setup_partial\n        test_hello.teardown = my_teardown_partial\n    ')
    result = pytester.runpytest(p, '-p', 'nose', '-Wignore::pytest.PytestRemovedIn8Warning')
    result.stdout.fnmatch_lines(['*2 passed*'])

def test_module_level_setup(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        from nose.tools import with_setup\n        items = {}\n\n        def setup():\n            items.setdefault("setup", []).append("up")\n\n        def teardown():\n            items.setdefault("setup", []).append("down")\n\n        def setup2():\n            items.setdefault("setup2", []).append("up")\n\n        def teardown2():\n            items.setdefault("setup2", []).append("down")\n\n        def test_setup_module_setup():\n            assert items["setup"] == ["up"]\n\n        def test_setup_module_setup_again():\n            assert items["setup"] == ["up"]\n\n        @with_setup(setup2, teardown2)\n        def test_local_setup():\n            assert items["setup"] == ["up"]\n            assert items["setup2"] == ["up"]\n\n        @with_setup(setup2, teardown2)\n        def test_local_setup_again():\n            assert items["setup"] == ["up"]\n            assert items["setup2"] == ["up", "down", "up"]\n    ')
    result = pytester.runpytest('-p', 'nose', '-Wignore::pytest.PytestRemovedIn8Warning')
    result.stdout.fnmatch_lines(['*4 passed*'])

def test_nose_style_setup_teardown(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile('\n        values = []\n\n        def setup_module():\n            values.append(1)\n\n        def teardown_module():\n            del values[0]\n\n        def test_hello():\n            assert values == [1]\n\n        def test_world():\n            assert values == [1]\n        ')
    result = pytester.runpytest('-p', 'nose')
    result.stdout.fnmatch_lines(['*2 passed*'])

def test_fixtures_nose_setup_issue8394(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile('\n        def setup_module():\n            pass\n\n        def teardown_module():\n            pass\n\n        def setup_function(func):\n            pass\n\n        def teardown_function(func):\n            pass\n\n        def test_world():\n            pass\n\n        class Test(object):\n            def setup_class(cls):\n                pass\n\n            def teardown_class(cls):\n                pass\n\n            def setup_method(self, meth):\n                pass\n\n            def teardown_method(self, meth):\n                pass\n\n            def test_method(self): pass\n        ')
    match = '*no docstring available*'
    result = pytester.runpytest('--fixtures')
    assert result.ret == 0
    result.stdout.no_fnmatch_line(match)
    result = pytester.runpytest('--fixtures', '-v')
    assert result.ret == 0
    result.stdout.fnmatch_lines([match, match, match, match])

def test_nose_setup_ordering(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        def setup_module(mod):\n            mod.visited = True\n\n        class TestClass(object):\n            def setup(self):\n                assert visited\n                self.visited_cls = True\n            def test_first(self):\n                assert visited\n                assert self.visited_cls\n        ')
    result = pytester.runpytest('-Wignore::pytest.PytestRemovedIn8Warning')
    result.stdout.fnmatch_lines(['*1 passed*'])

def test_apiwrapper_problem_issue260(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile("\n        import unittest\n        class TestCase(unittest.TestCase):\n            def setup(self):\n                #should not be called in unittest testcases\n                assert 0, 'setup'\n            def teardown(self):\n                #should not be called in unittest testcases\n                assert 0, 'teardown'\n            def setUp(self):\n                print('setup')\n            def tearDown(self):\n                print('teardown')\n            def test_fun(self):\n                pass\n        ")
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)

def test_setup_teardown_linking_issue265(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        import pytest\n\n        class TestGeneric(object):\n            def test_nothing(self):\n                """Tests the API of the implementation (for generic and specialized)."""\n\n        @pytest.mark.skipif("True", reason=\n                    "Skip tests to check if teardown is skipped as well.")\n        class TestSkipTeardown(TestGeneric):\n\n            def setup(self):\n                """Sets up my specialized implementation for $COOL_PLATFORM."""\n                raise Exception("should not call setup for skipped tests")\n\n            def teardown(self):\n                """Undoes the setup."""\n                raise Exception("should not call teardown for skipped tests")\n        ')
    reprec = pytester.runpytest()
    reprec.assert_outcomes(passed=1, skipped=1)

def test_SkipTest_during_collection(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    p = pytester.makepyfile('\n        import nose\n        raise nose.SkipTest("during collection")\n        def test_failing():\n            assert False\n        ')
    result = pytester.runpytest(p)
    result.assert_outcomes(skipped=1, warnings=0)

def test_SkipTest_in_test(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makepyfile('\n        import nose\n\n        def test_skipping():\n            raise nose.SkipTest("in test")\n        ')
    reprec = pytester.inline_run()
    reprec.assertoutcome(skipped=1)

def test_istest_function_decorator(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    p = pytester.makepyfile('\n        import nose.tools\n        @nose.tools.istest\n        def not_test_prefix():\n            pass\n        ')
    result = pytester.runpytest(p)
    result.assert_outcomes(passed=1)

def test_nottest_function_decorator(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile('\n        import nose.tools\n        @nose.tools.nottest\n        def test_prefix():\n            pass\n        ')
    reprec = pytester.inline_run()
    assert not reprec.getfailedcollections()
    calls = reprec.getreports('pytest_runtest_logreport')
    assert not calls

def test_istest_class_decorator(pytester: Pytester) -> None:
    if False:
        return 10
    p = pytester.makepyfile('\n        import nose.tools\n        @nose.tools.istest\n        class NotTestPrefix(object):\n            def test_method(self):\n                pass\n        ')
    result = pytester.runpytest(p)
    result.assert_outcomes(passed=1)

def test_nottest_class_decorator(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile('\n        import nose.tools\n        @nose.tools.nottest\n        class TestPrefix(object):\n            def test_method(self):\n                pass\n        ')
    reprec = pytester.inline_run()
    assert not reprec.getfailedcollections()
    calls = reprec.getreports('pytest_runtest_logreport')
    assert not calls

def test_skip_test_with_unicode(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile("        import unittest\n        class TestClass():\n            def test_io(self):\n                raise unittest.SkipTest('😊')\n        ")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['* 1 skipped *'])

def test_raises(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile('\n        from nose.tools import raises\n\n        @raises(RuntimeError)\n        def test_raises_runtimeerror():\n            raise RuntimeError\n\n        @raises(Exception)\n        def test_raises_baseexception_not_caught():\n            raise BaseException\n\n        @raises(BaseException)\n        def test_raises_baseexception_caught():\n            raise BaseException\n        ')
    result = pytester.runpytest('-vv')
    result.stdout.fnmatch_lines(['test_raises.py::test_raises_runtimeerror PASSED*', 'test_raises.py::test_raises_baseexception_not_caught FAILED*', 'test_raises.py::test_raises_baseexception_caught PASSED*', '*= FAILURES =*', '*_ test_raises_baseexception_not_caught _*', '', 'arg = (), kw = {}', '', '    def newfunc(*arg, **kw):', '        try:', '>           func(*arg, **kw)', '', '*/nose/*: ', '_ _ *', '', '    @raises(Exception)', '    def test_raises_baseexception_not_caught():', '>       raise BaseException', 'E       BaseException', '', 'test_raises.py:9: BaseException', '* 1 failed, 2 passed *'])

def test_nose_setup_skipped_if_non_callable(pytester: Pytester) -> None:
    if False:
        return 10
    'Regression test for #9391.'
    p = pytester.makepyfile(__init__='', setup='\n        ', teardown='\n        ', test_it='\n        from . import setup, teardown\n\n        def test_it():\n            pass\n        ')
    result = pytester.runpytest(p.parent, '-p', 'nose')
    assert result.ret == 0

@pytest.mark.parametrize('fixture_name', ('teardown', 'teardown_class'))
def test_teardown_fixture_not_called_directly(fixture_name, pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    'Regression test for #10597.'
    p = pytester.makepyfile(f'\n        import pytest\n\n        class TestHello:\n\n            @pytest.fixture\n            def {fixture_name}(self):\n                yield\n\n            def test_hello(self, {fixture_name}):\n                assert True\n        ')
    result = pytester.runpytest(p, '-p', 'nose')
    assert result.ret == 0