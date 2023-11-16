import sys
import pytest
from _pytest.config import ExitCode
from _pytest.pytester import Pytester

@pytest.fixture(params=['--setup-only', '--setup-plan', '--setup-show'], scope='module')
def mode(request):
    if False:
        return 10
    return request.param

def test_show_only_active_fixtures(pytester: Pytester, mode, dummy_yaml_custom_test) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def _arg0():\n            """hidden arg0 fixture"""\n        @pytest.fixture\n        def arg1():\n            """arg1 docstring"""\n        def test_arg1(arg1):\n            pass\n    ')
    result = pytester.runpytest(mode)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*SETUP    F arg1*', '*test_arg1 (fixtures used: arg1)*', '*TEARDOWN F arg1*'])
    result.stdout.no_fnmatch_line('*_arg0*')

def test_show_different_scopes(pytester: Pytester, mode) -> None:
    if False:
        i = 10
        return i + 15
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def arg_function():\n            """function scoped fixture"""\n        @pytest.fixture(scope=\'session\')\n        def arg_session():\n            """session scoped fixture"""\n        def test_arg1(arg_session, arg_function):\n            pass\n    ')
    result = pytester.runpytest(mode, p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['SETUP    S arg_session*', '*SETUP    F arg_function*', '*test_arg1 (fixtures used: arg_function, arg_session)*', '*TEARDOWN F arg_function*', 'TEARDOWN S arg_session*'])

def test_show_nested_fixtures(pytester: Pytester, mode) -> None:
    if False:
        return 10
    pytester.makeconftest('\n        import pytest\n        @pytest.fixture(scope=\'session\')\n        def arg_same():\n            """session scoped fixture"""\n        ')
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture(scope=\'function\')\n        def arg_same(arg_same):\n            """function scoped fixture"""\n        def test_arg1(arg_same):\n            pass\n    ')
    result = pytester.runpytest(mode, p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['SETUP    S arg_same*', '*SETUP    F arg_same (fixtures used: arg_same)*', '*test_arg1 (fixtures used: arg_same)*', '*TEARDOWN F arg_same*', 'TEARDOWN S arg_same*'])

def test_show_fixtures_with_autouse(pytester: Pytester, mode) -> None:
    if False:
        return 10
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def arg_function():\n            """function scoped fixture"""\n        @pytest.fixture(scope=\'session\', autouse=True)\n        def arg_session():\n            """session scoped fixture"""\n        def test_arg1(arg_function):\n            pass\n    ')
    result = pytester.runpytest(mode, p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['SETUP    S arg_session*', '*SETUP    F arg_function*', '*test_arg1 (fixtures used: arg_function, arg_session)*'])

def test_show_fixtures_with_parameters(pytester: Pytester, mode) -> None:
    if False:
        print('Hello World!')
    pytester.makeconftest('\n        import pytest\n        @pytest.fixture(scope=\'session\', params=[\'foo\', \'bar\'])\n        def arg_same():\n            """session scoped fixture"""\n        ')
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture(scope=\'function\')\n        def arg_other(arg_same):\n            """function scoped fixture"""\n        def test_arg1(arg_other):\n            pass\n    ')
    result = pytester.runpytest(mode, p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(["SETUP    S arg_same?'foo'?", "TEARDOWN S arg_same?'foo'?", "SETUP    S arg_same?'bar'?", "TEARDOWN S arg_same?'bar'?"])

def test_show_fixtures_with_parameter_ids(pytester: Pytester, mode) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makeconftest('\n        import pytest\n        @pytest.fixture(\n            scope=\'session\', params=[\'foo\', \'bar\'], ids=[\'spam\', \'ham\'])\n        def arg_same():\n            """session scoped fixture"""\n        ')
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture(scope=\'function\')\n        def arg_other(arg_same):\n            """function scoped fixture"""\n        def test_arg1(arg_other):\n            pass\n    ')
    result = pytester.runpytest(mode, p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(["SETUP    S arg_same?'spam'?", "SETUP    S arg_same?'ham'?"])

def test_show_fixtures_with_parameter_ids_function(pytester: Pytester, mode) -> None:
    if False:
        print('Hello World!')
    p = pytester.makepyfile("\n        import pytest\n        @pytest.fixture(params=['foo', 'bar'], ids=lambda p: p.upper())\n        def foobar():\n            pass\n        def test_foobar(foobar):\n            pass\n    ")
    result = pytester.runpytest(mode, p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*SETUP    F foobar?'FOO'?", "*SETUP    F foobar?'BAR'?"])

def test_dynamic_fixture_request(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    p = pytester.makepyfile("\n        import pytest\n        @pytest.fixture()\n        def dynamically_requested_fixture():\n            pass\n        @pytest.fixture()\n        def dependent_fixture(request):\n            request.getfixturevalue('dynamically_requested_fixture')\n        def test_dyn(dependent_fixture):\n            pass\n    ")
    result = pytester.runpytest('--setup-only', p)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*SETUP    F dynamically_requested_fixture', '*TEARDOWN F dynamically_requested_fixture'])

def test_capturing(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    p = pytester.makepyfile("\n        import pytest, sys\n        @pytest.fixture()\n        def one():\n            sys.stdout.write('this should be captured')\n            sys.stderr.write('this should also be captured')\n        @pytest.fixture()\n        def two(one):\n            assert 0\n        def test_capturing(two):\n            pass\n    ")
    result = pytester.runpytest('--setup-only', p)
    result.stdout.fnmatch_lines(['this should be captured', 'this should also be captured'])

def test_show_fixtures_and_execute_test(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    'Verify that setups are shown and tests are executed.'
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def arg():\n            assert True\n        def test_arg(arg):\n            assert False\n    ')
    result = pytester.runpytest('--setup-show', p)
    assert result.ret == 1
    result.stdout.fnmatch_lines(['*SETUP    F arg*', '*test_arg (fixtures used: arg)F*', '*TEARDOWN F arg*'])

def test_setup_show_with_KeyboardInterrupt_in_test(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    p = pytester.makepyfile('\n        import pytest\n        @pytest.fixture\n        def arg():\n            pass\n        def test_arg(arg):\n            raise KeyboardInterrupt()\n    ')
    result = pytester.runpytest('--setup-show', p, no_reraise_ctrlc=True)
    result.stdout.fnmatch_lines(['*SETUP    F arg*', '*test_arg (fixtures used: arg)*', '*TEARDOWN F arg*', '*! KeyboardInterrupt !*', '*= no tests ran in *'])
    assert result.ret == ExitCode.INTERRUPTED

def test_show_fixture_action_with_bytes(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    test_file = pytester.makepyfile("\n        import pytest\n\n        @pytest.mark.parametrize('data', [b'Hello World'])\n        def test_data(data):\n            pass\n        ")
    result = pytester.run(sys.executable, '-bb', '-m', 'pytest', '--setup-show', str(test_file))
    assert result.ret == 0