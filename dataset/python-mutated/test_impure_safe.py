from returns.io import IOSuccess, impure_safe

@impure_safe
def _function(number: int) -> float:
    if False:
        while True:
            i = 10
    return number / number

def test_safe_iosuccess():
    if False:
        i = 10
        return i + 15
    'Ensures that safe decorator works correctly for IOSuccess case.'
    assert _function(1) == IOSuccess(1.0)

def test_safe_iofailure():
    if False:
        return 10
    'Ensures that safe decorator works correctly for IOFailure case.'
    failed = _function(0)
    assert isinstance(failed.failure()._inner_value, ZeroDivisionError)