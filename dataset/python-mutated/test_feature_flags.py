from hypothesis import given, strategies as st
from hypothesis.strategies._internal.featureflags import FeatureFlags, FeatureStrategy
from tests.common.debug import find_any, minimal
STRAT = FeatureStrategy()

def test_can_all_be_enabled():
    if False:
        while True:
            i = 10
    find_any(STRAT, lambda x: all((x.is_enabled(i) for i in range(100))))

def test_minimizes_open():
    if False:
        i = 10
        return i + 15
    features = range(10)
    flags = minimal(STRAT, lambda x: [x.is_enabled(i) for i in features])
    assert all((flags.is_enabled(i) for i in features))

def test_minimizes_individual_features_to_open():
    if False:
        while True:
            i = 10
    features = list(range(10))
    flags = minimal(STRAT, lambda x: sum((x.is_enabled(i) for i in features)) < len(features))
    assert all((flags.is_enabled(i) for i in features[:-1]))
    assert not flags.is_enabled(features[-1])

def test_marks_unknown_features_as_enabled():
    if False:
        for i in range(10):
            print('nop')
    x = find_any(STRAT, lambda v: True)
    assert x.is_enabled('fish')

def test_by_default_all_enabled():
    if False:
        i = 10
        return i + 15
    f = FeatureFlags()
    assert f.is_enabled('foo')

def test_eval_featureflags_repr():
    if False:
        print('Hello World!')
    flags = FeatureFlags(enabled=['on'], disabled=['off'])
    assert flags.is_enabled('on')
    assert not flags.is_enabled('off')
    flags2 = eval(repr(flags))
    assert flags2.is_enabled('on')
    assert not flags2.is_enabled('off')

@given(st.data())
def test_repr_can_be_evalled(data):
    if False:
        for i in range(10):
            print('nop')
    flags = data.draw(STRAT)
    features = data.draw(st.lists(st.text(), unique=True))
    for f in features:
        flags.is_enabled(f)
    flags2 = eval(repr(flags))
    for f in features:
        assert flags2.is_enabled(f) == flags.is_enabled(f)
    more_features = data.draw(st.lists(st.text().filter(lambda s: s not in features)))
    for f in more_features:
        assert flags2.is_enabled(f)