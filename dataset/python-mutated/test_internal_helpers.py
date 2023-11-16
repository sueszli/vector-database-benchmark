import pytest
from hypothesis.internal.floats import is_negative

def test_is_negative_gives_good_type_error():
    if False:
        return 10
    x = 'foo'
    with pytest.raises(TypeError) as e:
        is_negative(x)
    assert repr(x) in e.value.args[0]