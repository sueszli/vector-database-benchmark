from copy import copy, deepcopy
import pytest
from returns.context import RequiresContextIOResult
from returns.primitives.exceptions import ImmutableStateError

def test_requires_context_result_immutable():
    if False:
        for i in range(10):
            print('nop')
    'Ensures that container is immutable.'
    with pytest.raises(ImmutableStateError):
        RequiresContextIOResult.from_value(1).abc = 1

def test_requires_context_result_immutable_copy():
    if False:
        i = 10
        return i + 15
    'Ensures that helper returns it self when passed to copy function.'
    context_ioresult = RequiresContextIOResult.from_value(1)
    assert context_ioresult is copy(context_ioresult)

def test_requires_context_result_immutable_deepcopy():
    if False:
        while True:
            i = 10
    'Ensures that helper returns it self when passed to deepcopy function.'
    requires_context = RequiresContextIOResult.from_value(1)
    assert requires_context is deepcopy(requires_context)