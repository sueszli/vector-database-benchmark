import pytest
from hypothesis import given, settings, strategies as st
from tests.common.utils import fails_with

def fails_with_output(expected, error=AssertionError, **kw):
    if False:
        return 10

    def _inner(f):
        if False:
            for i in range(10):
                print('nop')

        def _new():
            if False:
                return 10
            with pytest.raises(error) as err:
                settings(print_blob=False, derandomize=True, **kw)(f)()
            got = '\n'.join(err.value.__notes__).strip() + '\n'
            assert got == expected.strip() + '\n'
        return _new
    return _inner

@fails_with_output('\nFalsifying example: test_inquisitor_comments_basic_fail_if_either(\n    # The test always failed when commented parts were varied together.\n    a=False,  # or any other generated value\n    b=True,\n    c=[],  # or any other generated value\n    d=True,\n    e=False,  # or any other generated value\n)\n')
@given(st.booleans(), st.booleans(), st.lists(st.none()), st.booleans(), st.booleans())
def test_inquisitor_comments_basic_fail_if_either(a, b, c, d, e):
    if False:
        for i in range(10):
            print('nop')
    assert not (b and d)

@fails_with_output("\nFalsifying example: test_inquisitor_comments_basic_fail_if_not_all(\n    # The test sometimes passed when commented parts were varied together.\n    a='',  # or any other generated value\n    b='',  # or any other generated value\n    c='',  # or any other generated value\n)\n")
@given(st.text(), st.text(), st.text())
def test_inquisitor_comments_basic_fail_if_not_all(a, b, c):
    if False:
        for i in range(10):
            print('nop')
    condition = a and b and c
    assert condition

@fails_with_output("\nFalsifying example: test_inquisitor_no_together_comment_if_single_argument(\n    a='',\n    b='',  # or any other generated value\n)\n")
@given(st.text(), st.text())
def test_inquisitor_no_together_comment_if_single_argument(a, b):
    if False:
        while True:
            i = 10
    assert a

@fails_with(ZeroDivisionError)
@settings(database=None)
@given(start_date=st.datetimes(), data=st.data())
def test_issue_3755_regression(start_date, data):
    if False:
        for i in range(10):
            print('nop')
    data.draw(st.datetimes(min_value=start_date))
    raise ZeroDivisionError