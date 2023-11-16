""""
Test module for testing ``pandas._testing.assert_produces_warning``.
"""
import warnings
import pytest
from pandas.errors import DtypeWarning, PerformanceWarning
import pandas._testing as tm

@pytest.fixture(params=[RuntimeWarning, ResourceWarning, UserWarning, FutureWarning, DeprecationWarning, PerformanceWarning, DtypeWarning])
def category(request):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return unique warning.\n\n    Useful for testing behavior of tm.assert_produces_warning with various categories.\n    '
    return request.param

@pytest.fixture(params=[(RuntimeWarning, UserWarning), (UserWarning, FutureWarning), (FutureWarning, RuntimeWarning), (DeprecationWarning, PerformanceWarning), (PerformanceWarning, FutureWarning), (DtypeWarning, DeprecationWarning), (ResourceWarning, DeprecationWarning), (FutureWarning, DeprecationWarning)], ids=lambda x: type(x).__name__)
def pair_different_warnings(request):
    if False:
        i = 10
        return i + 15
    '\n    Return pair or different warnings.\n\n    Useful for testing how several different warnings are handled\n    in tm.assert_produces_warning.\n    '
    return request.param

def f():
    if False:
        return 10
    warnings.warn('f1', FutureWarning)
    warnings.warn('f2', RuntimeWarning)

@pytest.mark.filterwarnings('ignore:f1:FutureWarning')
def test_assert_produces_warning_honors_filter():
    if False:
        i = 10
        return i + 15
    msg = 'Caused unexpected warning\\(s\\)'
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(RuntimeWarning):
            f()
    with tm.assert_produces_warning(RuntimeWarning, raise_on_extra_warnings=False):
        f()

@pytest.mark.parametrize('message, match', [('', None), ('', ''), ('Warning message', '.*'), ('Warning message', 'War'), ('Warning message', '[Ww]arning'), ('Warning message', 'age'), ('Warning message', 'age$'), ('Message 12-234 with numbers', '\\d{2}-\\d{3}'), ('Message 12-234 with numbers', '^Mes.*\\d{2}-\\d{3}'), ('Message 12-234 with numbers', '\\d{2}-\\d{3}\\s\\S+'), ('Message, which we do not match', None)])
def test_catch_warning_category_and_match(category, message, match):
    if False:
        print('Hello World!')
    with tm.assert_produces_warning(category, match=match):
        warnings.warn(message, category)

def test_fail_to_match_runtime_warning():
    if False:
        return 10
    category = RuntimeWarning
    match = 'Did not see this warning'
    unmatched = "Did not see warning 'RuntimeWarning' matching 'Did not see this warning'. The emitted warning messages are \\[RuntimeWarning\\('This is not a match.'\\), RuntimeWarning\\('Another unmatched warning.'\\)\\]"
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn('This is not a match.', category)
            warnings.warn('Another unmatched warning.', category)

def test_fail_to_match_future_warning():
    if False:
        for i in range(10):
            print('nop')
    category = FutureWarning
    match = 'Warning'
    unmatched = "Did not see warning 'FutureWarning' matching 'Warning'. The emitted warning messages are \\[FutureWarning\\('This is not a match.'\\), FutureWarning\\('Another unmatched warning.'\\)\\]"
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn('This is not a match.', category)
            warnings.warn('Another unmatched warning.', category)

def test_fail_to_match_resource_warning():
    if False:
        return 10
    category = ResourceWarning
    match = '\\d+'
    unmatched = "Did not see warning 'ResourceWarning' matching '\\\\d\\+'. The emitted warning messages are \\[ResourceWarning\\('This is not a match.'\\), ResourceWarning\\('Another unmatched warning.'\\)\\]"
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn('This is not a match.', category)
            warnings.warn('Another unmatched warning.', category)

def test_fail_to_catch_actual_warning(pair_different_warnings):
    if False:
        return 10
    (expected_category, actual_category) = pair_different_warnings
    match = 'Did not see expected warning of class'
    with pytest.raises(AssertionError, match=match):
        with tm.assert_produces_warning(expected_category):
            warnings.warn('warning message', actual_category)

def test_ignore_extra_warning(pair_different_warnings):
    if False:
        i = 10
        return i + 15
    (expected_category, extra_category) = pair_different_warnings
    with tm.assert_produces_warning(expected_category, raise_on_extra_warnings=False):
        warnings.warn('Expected warning', expected_category)
        warnings.warn('Unexpected warning OK', extra_category)

def test_raise_on_extra_warning(pair_different_warnings):
    if False:
        while True:
            i = 10
    (expected_category, extra_category) = pair_different_warnings
    match = 'Caused unexpected warning\\(s\\)'
    with pytest.raises(AssertionError, match=match):
        with tm.assert_produces_warning(expected_category):
            warnings.warn('Expected warning', expected_category)
            warnings.warn('Unexpected warning NOT OK', extra_category)

def test_same_category_different_messages_first_match():
    if False:
        print('Hello World!')
    category = UserWarning
    with tm.assert_produces_warning(category, match='^Match this'):
        warnings.warn('Match this', category)
        warnings.warn('Do not match that', category)
        warnings.warn('Do not match that either', category)

def test_same_category_different_messages_last_match():
    if False:
        i = 10
        return i + 15
    category = DeprecationWarning
    with tm.assert_produces_warning(category, match='^Match this'):
        warnings.warn('Do not match that', category)
        warnings.warn('Do not match that either', category)
        warnings.warn('Match this', category)

def test_match_multiple_warnings():
    if False:
        while True:
            i = 10
    category = (FutureWarning, UserWarning)
    with tm.assert_produces_warning(category, match='^Match this'):
        warnings.warn('Match this', FutureWarning)
        warnings.warn('Match this too', UserWarning)

def test_right_category_wrong_match_raises(pair_different_warnings):
    if False:
        print('Hello World!')
    (target_category, other_category) = pair_different_warnings
    with pytest.raises(AssertionError, match='Did not see warning.*matching'):
        with tm.assert_produces_warning(target_category, match='^Match this'):
            warnings.warn('Do not match it', target_category)
            warnings.warn('Match this', other_category)

@pytest.mark.parametrize('false_or_none', [False, None])
class TestFalseOrNoneExpectedWarning:

    def test_raise_on_warning(self, false_or_none):
        if False:
            print('Hello World!')
        msg = 'Caused unexpected warning\\(s\\)'
        with pytest.raises(AssertionError, match=msg):
            with tm.assert_produces_warning(false_or_none):
                f()

    def test_no_raise_without_warning(self, false_or_none):
        if False:
            print('Hello World!')
        with tm.assert_produces_warning(false_or_none):
            pass

    def test_no_raise_with_false_raise_on_extra(self, false_or_none):
        if False:
            i = 10
            return i + 15
        with tm.assert_produces_warning(false_or_none, raise_on_extra_warnings=False):
            f()

def test_raises_during_exception():
    if False:
        i = 10
        return i + 15
    msg = "Did not see expected warning of class 'UserWarning'"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(UserWarning):
            raise ValueError
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(UserWarning):
            warnings.warn('FutureWarning', FutureWarning)
            raise IndexError
    msg = 'Caused unexpected warning'
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(None):
            warnings.warn('FutureWarning', FutureWarning)
            raise SystemError

def test_passes_during_exception():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(SyntaxError, match='Error'):
        with tm.assert_produces_warning(None):
            raise SyntaxError('Error')
    with pytest.raises(ValueError, match='Error'):
        with tm.assert_produces_warning(FutureWarning, match='FutureWarning'):
            warnings.warn('FutureWarning', FutureWarning)
            raise ValueError('Error')