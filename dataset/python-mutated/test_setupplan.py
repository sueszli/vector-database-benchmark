from _pytest.pytester import Pytester

def test_show_fixtures_and_test(pytester: Pytester, dummy_yaml_custom_test: None) -> None:
    if False:
        while True:
            i = 10
    'Verify that fixtures are not executed.'
    pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def arg():\n            assert False\n        def test_arg(arg):\n            assert False\n    ')
    result = pytester.runpytest('--setup-plan')
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*SETUP    F arg*', '*test_arg (fixtures used: arg)', '*TEARDOWN F arg*'])

def test_show_multi_test_fixture_setup_and_teardown_correctly_simple(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    'Verify that when a fixture lives for longer than a single test, --setup-plan\n    correctly displays the SETUP/TEARDOWN indicators the right number of times.\n\n    As reported in https://github.com/pytest-dev/pytest/issues/2049\n    --setup-plan was showing SETUP/TEARDOWN on every test, even when the fixture\n    should persist through multiple tests.\n\n    (Note that this bug never affected actual test execution, which used the\n    correct fixture lifetimes. It was purely a display bug for --setup-plan, and\n    did not affect the related --setup-show or --setup-only.)\n    '
    pytester.makepyfile("\n        import pytest\n        @pytest.fixture(scope = 'class')\n        def fix():\n            return object()\n        class TestClass:\n            def test_one(self, fix):\n                assert False\n            def test_two(self, fix):\n                assert False\n    ")
    result = pytester.runpytest('--setup-plan')
    assert result.ret == 0
    setup_fragment = 'SETUP    C fix'
    setup_count = 0
    teardown_fragment = 'TEARDOWN C fix'
    teardown_count = 0
    for line in result.stdout.lines:
        if setup_fragment in line:
            setup_count += 1
        if teardown_fragment in line:
            teardown_count += 1
    assert setup_count == 1
    assert teardown_count == 1

def test_show_multi_test_fixture_setup_and_teardown_same_as_setup_show(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    'Verify that SETUP/TEARDOWN messages match what comes out of --setup-show.'
    pytester.makepyfile("\n        import pytest\n        @pytest.fixture(scope = 'session')\n        def sess():\n            return True\n        @pytest.fixture(scope = 'module')\n        def mod():\n            return True\n        @pytest.fixture(scope = 'class')\n        def cls():\n            return True\n        @pytest.fixture(scope = 'function')\n        def func():\n            return True\n        def test_outside(sess, mod, cls, func):\n            assert True\n        class TestCls:\n            def test_one(self, sess, mod, cls, func):\n                assert True\n            def test_two(self, sess, mod, cls, func):\n                assert True\n    ")
    plan_result = pytester.runpytest('--setup-plan')
    show_result = pytester.runpytest('--setup-show')
    plan_lines = [line for line in plan_result.stdout.lines if 'SETUP' in line or 'TEARDOWN' in line]
    show_lines = [line for line in show_result.stdout.lines if 'SETUP' in line or 'TEARDOWN' in line]
    assert plan_lines == show_lines