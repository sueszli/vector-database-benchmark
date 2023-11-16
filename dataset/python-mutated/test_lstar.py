import hypothesis.strategies as st
from hypothesis import Phase, example, given, settings
from hypothesis.internal.conjecture.dfa.lstar import LStar

@st.composite
def byte_order(draw):
    if False:
        while True:
            i = 10
    ls = draw(st.permutations(range(256)))
    n = draw(st.integers(0, len(ls)))
    return ls[:n]

@example({0}, [1])
@given(st.sets(st.integers(0, 255)), byte_order())
@settings(phases=set(settings.default.phases) - {Phase.target})
def test_learning_always_changes_generation(chars, order):
    if False:
        while True:
            i = 10
    learner = LStar(lambda s: len(s) == 1 and s[0] in chars)
    for c in order:
        prev = learner.generation
        s = bytes([c])
        if learner.dfa.matches(s) != learner.member(s):
            learner.learn(s)
            assert learner.generation > prev