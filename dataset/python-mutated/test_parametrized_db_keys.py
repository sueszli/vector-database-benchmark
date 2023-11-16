import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
DB_KEY_TESTCASE = '\nfrom hypothesis import settings, given\nfrom hypothesis.database import InMemoryExampleDatabase\nfrom hypothesis.strategies import booleans\nimport pytest\n\nDB = InMemoryExampleDatabase()\n\n\n@settings(database=DB)\n@given(booleans())\n@pytest.mark.parametrize("hi", (1, 2, 3))\n@pytest.mark.xfail()\ndef test_dummy_for_parametrized_db_keys(hi, i):\n    assert Fail  # Test *must* fail for it to end up the database anyway\n\n\ndef test_DB_keys_for_parametrized_test():\n    assert len(DB.data) == 6\n'

def test_db_keys_for_parametrized_tests_are_unique(testdir):
    if False:
        print('Hello World!')
    script = testdir.makepyfile(DB_KEY_TESTCASE)
    testdir.runpytest(script).assert_outcomes(xfailed=3, passed=1)

@pytest.fixture(params=['a', 'b'])
def fixt(request):
    if False:
        for i in range(10):
            print('nop')
    return request.param

class TestNoDifferingExecutorsHealthCheck:

    @given(x=st.text())
    @pytest.mark.parametrize('i', range(2))
    def test_method(self, x, i):
        if False:
            i = 10
            return i + 15
        pass

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(x=st.text())
    def test_method_fixture(self, x, fixt):
        if False:
            print('Hello World!')
        pass