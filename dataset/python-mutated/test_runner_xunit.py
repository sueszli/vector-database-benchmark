"""Test correct setup/teardowns at module, class, and instance level."""
from typing import List
import pytest
from _pytest.pytester import Pytester

def test_module_and_function_setup(pytester: Pytester) -> None:
    if False:
        return 10
    reprec = pytester.inline_runsource("\n        modlevel = []\n        def setup_module(module):\n            assert not modlevel\n            module.modlevel.append(42)\n\n        def teardown_module(module):\n            modlevel.pop()\n\n        def setup_function(function):\n            function.answer = 17\n\n        def teardown_function(function):\n            del function.answer\n\n        def test_modlevel():\n            assert modlevel[0] == 42\n            assert test_modlevel.answer == 17\n\n        class TestFromClass(object):\n            def test_module(self):\n                assert modlevel[0] == 42\n                assert not hasattr(test_modlevel, 'answer')\n    ")
    rep = reprec.matchreport('test_modlevel')
    assert rep.passed
    rep = reprec.matchreport('test_module')
    assert rep.passed

def test_module_setup_failure_no_teardown(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    reprec = pytester.inline_runsource('\n        values = []\n        def setup_module(module):\n            values.append(1)\n            0/0\n\n        def test_nothing():\n            pass\n\n        def teardown_module(module):\n            values.append(2)\n    ')
    reprec.assertoutcome(failed=1)
    calls = reprec.getcalls('pytest_runtest_setup')
    assert calls[0].item.module.values == [1]

def test_setup_function_failure_no_teardown(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    reprec = pytester.inline_runsource('\n        modlevel = []\n        def setup_function(function):\n            modlevel.append(1)\n            0/0\n\n        def teardown_function(module):\n            modlevel.append(2)\n\n        def test_func():\n            pass\n    ')
    calls = reprec.getcalls('pytest_runtest_setup')
    assert calls[0].item.module.modlevel == [1]

def test_class_setup(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    reprec = pytester.inline_runsource('\n        class TestSimpleClassSetup(object):\n            clslevel = []\n            def setup_class(cls):\n                cls.clslevel.append(23)\n\n            def teardown_class(cls):\n                cls.clslevel.pop()\n\n            def test_classlevel(self):\n                assert self.clslevel[0] == 23\n\n        class TestInheritedClassSetupStillWorks(TestSimpleClassSetup):\n            def test_classlevel_anothertime(self):\n                assert self.clslevel == [23]\n\n        def test_cleanup():\n            assert not TestSimpleClassSetup.clslevel\n            assert not TestInheritedClassSetupStillWorks.clslevel\n    ')
    reprec.assertoutcome(passed=1 + 2 + 1)

def test_class_setup_failure_no_teardown(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    reprec = pytester.inline_runsource('\n        class TestSimpleClassSetup(object):\n            clslevel = []\n            def setup_class(cls):\n                0/0\n\n            def teardown_class(cls):\n                cls.clslevel.append(1)\n\n            def test_classlevel(self):\n                pass\n\n        def test_cleanup():\n            assert not TestSimpleClassSetup.clslevel\n    ')
    reprec.assertoutcome(failed=1, passed=1)

def test_method_setup(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    reprec = pytester.inline_runsource('\n        class TestSetupMethod(object):\n            def setup_method(self, meth):\n                self.methsetup = meth\n            def teardown_method(self, meth):\n                del self.methsetup\n\n            def test_some(self):\n                assert self.methsetup == self.test_some\n\n            def test_other(self):\n                assert self.methsetup == self.test_other\n    ')
    reprec.assertoutcome(passed=2)

def test_method_setup_failure_no_teardown(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    reprec = pytester.inline_runsource('\n        class TestMethodSetup(object):\n            clslevel = []\n            def setup_method(self, method):\n                self.clslevel.append(1)\n                0/0\n\n            def teardown_method(self, method):\n                self.clslevel.append(2)\n\n            def test_method(self):\n                pass\n\n        def test_cleanup():\n            assert TestMethodSetup.clslevel == [1]\n    ')
    reprec.assertoutcome(failed=1, passed=1)

def test_method_setup_uses_fresh_instances(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    reprec = pytester.inline_runsource('\n        class TestSelfState1(object):\n            memory = []\n            def test_hello(self):\n                self.memory.append(self)\n\n            def test_afterhello(self):\n                assert self != self.memory[0]\n    ')
    reprec.assertoutcome(passed=2, failed=0)

def test_setup_that_skips_calledagain(pytester: Pytester) -> None:
    if False:
        return 10
    p = pytester.makepyfile('\n        import pytest\n        def setup_module(mod):\n            pytest.skip("x")\n        def test_function1():\n            pass\n        def test_function2():\n            pass\n    ')
    reprec = pytester.inline_run(p)
    reprec.assertoutcome(skipped=2)

def test_setup_fails_again_on_all_tests(pytester: Pytester) -> None:
    if False:
        return 10
    p = pytester.makepyfile('\n        import pytest\n        def setup_module(mod):\n            raise ValueError(42)\n        def test_function1():\n            pass\n        def test_function2():\n            pass\n    ')
    reprec = pytester.inline_run(p)
    reprec.assertoutcome(failed=2)

def test_setup_funcarg_setup_when_outer_scope_fails(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    p = pytester.makepyfile('\n        import pytest\n        def setup_module(mod):\n            raise ValueError(42)\n        @pytest.fixture\n        def hello(request):\n            raise ValueError("xyz43")\n        def test_function1(hello):\n            pass\n        def test_function2(hello):\n            pass\n    ')
    result = pytester.runpytest(p)
    result.stdout.fnmatch_lines(['*function1*', '*ValueError*42*', '*function2*', '*ValueError*42*', '*2 errors*'])
    result.stdout.no_fnmatch_line('*xyz43*')

@pytest.mark.parametrize('arg', ['', 'arg'])
def test_setup_teardown_function_level_with_optional_argument(pytester: Pytester, monkeypatch, arg: str) -> None:
    if False:
        print('Hello World!')
    'Parameter to setup/teardown xunit-style functions parameter is now optional (#1728).'
    import sys
    trace_setups_teardowns: List[str] = []
    monkeypatch.setattr(sys, 'trace_setups_teardowns', trace_setups_teardowns, raising=False)
    p = pytester.makepyfile("\n        import pytest\n        import sys\n\n        trace = sys.trace_setups_teardowns.append\n\n        def setup_module({arg}): trace('setup_module')\n        def teardown_module({arg}): trace('teardown_module')\n\n        def setup_function({arg}): trace('setup_function')\n        def teardown_function({arg}): trace('teardown_function')\n\n        def test_function_1(): pass\n        def test_function_2(): pass\n\n        class Test(object):\n            def setup_method(self, {arg}): trace('setup_method')\n            def teardown_method(self, {arg}): trace('teardown_method')\n\n            def test_method_1(self): pass\n            def test_method_2(self): pass\n    ".format(arg=arg))
    result = pytester.inline_run(p)
    result.assertoutcome(passed=4)
    expected = ['setup_module', 'setup_function', 'teardown_function', 'setup_function', 'teardown_function', 'setup_method', 'teardown_method', 'setup_method', 'teardown_method', 'teardown_module']
    assert trace_setups_teardowns == expected