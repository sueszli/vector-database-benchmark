import pytest
from hypothesis import Phase, example, given, settings, strategies as st
OUTPUT_WITH_BREAK = '\nFalsifying explicit example: test(\n    x={0!r},\n    y={0!r},\n)\n'

@pytest.mark.parametrize('n', [10, 100])
def test_inserts_line_breaks_only_at_appropriate_lengths(n):
    if False:
        i = 10
        return i + 15

    @example('0' * n, '0' * n)
    @given(st.text(), st.text())
    def test(x, y):
        if False:
            i = 10
            return i + 15
        assert x < y
    with pytest.raises(AssertionError) as err:
        test()
    assert OUTPUT_WITH_BREAK.format('0' * n).strip() == '\n'.join(err.value.__notes__)

@given(kw=st.none())
def generate_phase(*args, kw):
    if False:
        for i in range(10):
            print('nop')
    assert args != (1, 2, 3)

@given(kw=st.none())
@example(kw=None)
@settings(phases=[Phase.explicit])
def explicit_phase(*args, kw):
    if False:
        for i in range(10):
            print('nop')
    assert args != (1, 2, 3)

@pytest.mark.parametrize('fn', [generate_phase, explicit_phase], ids=lambda fn: fn.__name__)
def test_vararg_output(fn):
    if False:
        while True:
            i = 10
    with pytest.raises(AssertionError) as err:
        fn(1, 2, 3)
    assert '1,\n    2,\n    3,\n' in '\n'.join(err.value.__notes__)