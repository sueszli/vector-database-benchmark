"""
PYTEST_DONT_REWRITE
"""
import difflib
import pprint
import pytest
from dirty_equals import HasRepr
from pydantic import BaseModel, ValidationError, field_validator

def _pformat_lines(obj):
    if False:
        i = 10
        return i + 15
    return pprint.pformat(obj).splitlines(keepends=True)

def _assert_eq(left, right):
    if False:
        print('Hello World!')
    if left != right:
        pytest.fail('\n' + '\n'.join(difflib.ndiff(_pformat_lines(left), _pformat_lines(right))))

def test_assert_raises_validation_error():
    if False:
        return 10

    class Model(BaseModel):
        a: str

        @field_validator('a')
        @classmethod
        def check_a(cls, v):
            if False:
                return 10
            assert v == 'a', 'invalid a'
            return v
    assert Model(a='a').a == 'a'
    with pytest.raises(ValidationError) as exc_info:
        Model(a='snap')
    _assert_eq([{'ctx': {'error': HasRepr(repr(AssertionError('invalid a')))}, 'input': 'snap', 'loc': ('a',), 'msg': 'Assertion failed, invalid a', 'type': 'assertion_error'}], exc_info.value.errors(include_url=False))