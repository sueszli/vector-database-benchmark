import pytest
from hypothesis import given
from hypothesis.errors import Flaky
from hypothesis.strategies import composite, integers, none

@pytest.mark.parametrize('e', [KeyboardInterrupt, SystemExit, GeneratorExit, ValueError])
def test_exception_propagates_fine(e):
    if False:
        i = 10
        return i + 15

    @given(integers())
    def test_raise(x):
        if False:
            i = 10
            return i + 15
        raise e
    with pytest.raises(e):
        test_raise()

@pytest.mark.parametrize('e', [KeyboardInterrupt, SystemExit, GeneratorExit, ValueError])
def test_exception_propagates_fine_from_strategy(e):
    if False:
        i = 10
        return i + 15

    @composite
    def interrupt_eventually(draw):
        if False:
            print('Hello World!')
        raise e
        return draw(none())

    @given(interrupt_eventually())
    def test_do_nothing(x):
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(e):
        test_do_nothing()

@pytest.mark.parametrize('e', [KeyboardInterrupt, ValueError])
def test_baseexception_no_rerun_no_flaky(e):
    if False:
        while True:
            i = 10
    runs = [0]
    interrupt = 3

    @given(integers())
    def test_raise_baseexception(x):
        if False:
            return 10
        runs[0] += 1
        if runs[0] == interrupt:
            raise e
    if issubclass(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        with pytest.raises(e):
            test_raise_baseexception()
        assert runs[0] == interrupt
    else:
        with pytest.raises(Flaky):
            test_raise_baseexception()

@pytest.mark.parametrize('e', [KeyboardInterrupt, SystemExit, GeneratorExit, ValueError])
def test_baseexception_in_strategy_no_rerun_no_flaky(e):
    if False:
        return 10
    runs = 0
    interrupt = 3

    @composite
    def interrupt_eventually(draw):
        if False:
            print('Hello World!')
        nonlocal runs
        runs += 1
        if runs == interrupt:
            raise e
        return draw(integers())

    @given(interrupt_eventually())
    def test_do_nothing(x):
        if False:
            i = 10
            return i + 15
        pass
    if issubclass(e, KeyboardInterrupt):
        with pytest.raises(e):
            test_do_nothing()
        assert runs == interrupt
    else:
        with pytest.raises(Flaky):
            test_do_nothing()
TEMPLATE = '\nfrom hypothesis import given, note, strategies as st\n\n@st.composite\ndef things(draw):\n    raise {exception}\n    # this line will not be executed, but must be here\n    # to pass draw function static reference check\n    return draw(st.none())\n\n\n@given(st.data(), st.integers())\ndef test(data, x):\n    if x > 100:\n        data.draw({strategy})\n        raise {exception}\n'

@pytest.mark.parametrize('exc_name', ['SystemExit', 'GeneratorExit'])
@pytest.mark.parametrize('use_composite', [True, False])
def test_explanations(testdir, exc_name, use_composite):
    if False:
        while True:
            i = 10
    code = TEMPLATE.format(exception=exc_name, strategy='things()' if use_composite else 'st.none()')
    test_file = str(testdir.makepyfile(code))
    pytest_stdout = str(testdir.runpytest_inprocess(test_file, '--tb=native').stdout)
    assert 'x=101' in pytest_stdout
    assert exc_name in pytest_stdout