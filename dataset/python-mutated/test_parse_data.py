"""
A "meta test" which tests the parsing of .test files. This is not meant to become exhaustive
but to ensure we maintain a basic level of ergonomics for mypy contributors.
"""
from mypy.test.helpers import Suite
from mypy.test.meta._pytest import PytestResult, run_pytest_data_suite

def _run_pytest(data_suite: str) -> PytestResult:
    if False:
        return 10
    return run_pytest_data_suite(data_suite, extra_args=[], max_attempts=1)

class ParseTestDataSuite(Suite):

    def test_parse_invalid_case(self) -> None:
        if False:
            return 10
        result = _run_pytest('\n            [case abc]\n            s: str\n            [case foo-XFAIL]\n            s: str\n            ')
        assert "Invalid testcase id 'foo-XFAIL'" in result.stdout

    def test_parse_invalid_section(self) -> None:
        if False:
            while True:
                i = 10
        result = _run_pytest('\n            [case abc]\n            s: str\n            [unknownsection]\n            abc\n            ')
        expected_lineno = result.input.splitlines().index('[unknownsection]') + 1
        expected = f".test:{expected_lineno}: Invalid section header [unknownsection] in case 'abc'"
        assert expected in result.stdout

    def test_bad_ge_version_check(self) -> None:
        if False:
            while True:
                i = 10
        actual = _run_pytest('\n            [case abc]\n            s: str\n            [out version>=3.8]\n            abc\n            ')
        assert 'version>=3.8 always true since minimum runtime version is (3, 8)' in actual.stdout

    def test_bad_eq_version_check(self) -> None:
        if False:
            return 10
        actual = _run_pytest('\n            [case abc]\n            s: str\n            [out version==3.7]\n            abc\n            ')
        assert 'version==3.7 always false since minimum runtime version is (3, 8)' in actual.stdout