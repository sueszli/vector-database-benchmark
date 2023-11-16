import pytest
from hypothesis import Phase, given, seed, settings, strategies as st, target
pytest_plugins = 'pytester'
TESTSUITE = '\nfrom hypothesis import given, strategies as st, target\n\n@given(st.integers(min_value=0))\ndef test_threshold_problem(x):\n    target(float(x))\n    {0}target(float(x * 2), label="double")\n    {0}assert x <= 100000\n    assert x <= 100\n'

@pytest.mark.parametrize('multiple', [False, True])
def test_reports_target_results(testdir, multiple):
    if False:
        return 10
    script = testdir.makepyfile(TESTSUITE.format('' if multiple else '# '))
    result = testdir.runpytest(script, '--tb=native', '-rN')
    out = '\n'.join(result.stdout.lines)
    assert 'Falsifying example' in out
    assert 'x=101' in out
    assert out.count('Highest target score') == 1
    assert result.ret != 0

def test_targeting_increases_max_length():
    if False:
        i = 10
        return i + 15
    strat = st.lists(st.booleans())

    @settings(database=None, max_examples=200, phases=[Phase.generate, Phase.target])
    @given(strat)
    def test_with_targeting(ls):
        if False:
            print('Hello World!')
        target(float(len(ls)))
        assert len(ls) <= 80
    with pytest.raises(AssertionError):
        test_with_targeting()

@given(st.integers(), st.integers())
def test_target_returns_value(a, b):
    if False:
        while True:
            i = 10
    difference = target(abs(a - b))
    assert difference == abs(a - b)
    assert isinstance(difference, int)

def test_targeting_can_be_disabled():
    if False:
        while True:
            i = 10
    strat = st.lists(st.integers(0, 255))

    def score(enabled):
        if False:
            i = 10
            return i + 15
        result = [0]
        phases = [Phase.generate]
        if enabled:
            phases.append(Phase.target)

        @seed(0)
        @settings(database=None, max_examples=200, phases=phases)
        @given(strat)
        def test(ls):
            if False:
                for i in range(10):
                    print('nop')
            score = float(sum(ls))
            result[0] = max(result[0], score)
            target(score)
        test()
        return result[0]
    assert score(enabled=True) > score(enabled=False)

def test_issue_2395_regression():
    if False:
        i = 10
        return i + 15

    @given(d=st.floats().filter(lambda x: abs(x) < 1000))
    @settings(max_examples=1000, database=None)
    @seed(93962505385993024185959759429298090872)
    def test_targeting_square_loss(d):
        if False:
            i = 10
            return i + 15
        target(-(d - 42.5) ** 2.0)
    test_targeting_square_loss()