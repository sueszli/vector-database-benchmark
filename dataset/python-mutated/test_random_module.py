import gc
import random
import pytest
from hypothesis import Phase, core, find, given, register_random, settings, strategies as st
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.internal import entropy
from hypothesis.internal.compat import GRAALPY, PYPY
from hypothesis.internal.entropy import deterministic_PRNG

def gc_collect():
    if False:
        for i in range(10):
            print('nop')
    if PYPY or GRAALPY:
        gc.collect()

def test_can_seed_random():
    if False:
        while True:
            i = 10

    @settings(phases=(Phase.generate, Phase.shrink))
    @given(st.random_module())
    def test(r):
        if False:
            i = 10
            return i + 15
        raise AssertionError
    with pytest.raises(AssertionError) as err:
        test()
    assert 'RandomSeeder(0)' in '\n'.join(err.value.__notes__)

@given(st.random_module(), st.random_module())
def test_seed_random_twice(r, r2):
    if False:
        while True:
            i = 10
    assert repr(r) == repr(r2)

@given(st.random_module())
def test_does_not_fail_health_check_if_randomness_is_used(r):
    if False:
        while True:
            i = 10
    random.getrandbits(128)

def test_cannot_register_non_Random():
    if False:
        i = 10
        return i + 15
    with pytest.raises(InvalidArgument):
        register_random('not a Random instance')

@pytest.mark.filterwarnings('ignore:It looks like `register_random` was passed an object that could be garbage collected')
def test_registering_a_Random_is_idempotent():
    if False:
        return 10
    gc_collect()
    n_registered = len(entropy.RANDOMS_TO_MANAGE)
    r = random.Random()
    register_random(r)
    register_random(r)
    assert len(entropy.RANDOMS_TO_MANAGE) == n_registered + 1
    del r
    gc_collect()
    assert len(entropy.RANDOMS_TO_MANAGE) == n_registered

def test_manages_registered_Random_instance():
    if False:
        i = 10
        return i + 15
    r = random.Random()
    register_random(r)
    state = r.getstate()
    result = []

    @given(st.integers())
    def inner(x):
        if False:
            return 10
        v = r.random()
        if result:
            assert v == result[0]
        else:
            result.append(v)
    inner()
    assert state == r.getstate()

def test_registered_Random_is_seeded_by_random_module_strategy():
    if False:
        print('Hello World!')
    r = random.Random()
    register_random(r)
    state = r.getstate()
    results = set()
    count = [0]

    @given(st.integers())
    def inner(x):
        if False:
            return 10
        results.add(r.random())
        count[0] += 1
    inner()
    assert count[0] > len(results) * 0.9, 'too few unique random numbers'
    assert state == r.getstate()

@given(st.random_module())
def test_will_actually_use_the_random_seed(rnd):
    if False:
        for i in range(10):
            print('nop')
    a = random.randint(0, 100)
    b = random.randint(0, 100)
    random.seed(rnd.seed)
    assert a == random.randint(0, 100)
    assert b == random.randint(0, 100)

def test_given_does_not_pollute_state():
    if False:
        i = 10
        return i + 15
    with deterministic_PRNG():

        @given(st.random_module())
        def test(r):
            if False:
                i = 10
                return i + 15
            pass
        test()
        state_a = random.getstate()
        state_a2 = core._hypothesis_global_random.getstate()
        test()
        state_b = random.getstate()
        state_b2 = core._hypothesis_global_random.getstate()
        assert state_a == state_b
        assert state_a2 != state_b2

def test_find_does_not_pollute_state():
    if False:
        return 10
    with deterministic_PRNG():
        find(st.random_module(), lambda r: True)
        state_a = random.getstate()
        state_a2 = core._hypothesis_global_random.getstate()
        find(st.random_module(), lambda r: True)
        state_b = random.getstate()
        state_b2 = core._hypothesis_global_random.getstate()
        assert state_a == state_b
        assert state_a2 != state_b2

@pytest.mark.filterwarnings('ignore:It looks like `register_random` was passed an object that could be garbage collected')
def test_evil_prng_registration_nonsense():
    if False:
        while True:
            i = 10
    gc_collect()
    n_registered = len(entropy.RANDOMS_TO_MANAGE)
    (r1, r2, r3) = (random.Random(1), random.Random(2), random.Random(3))
    s2 = r2.getstate()
    register_random(r1)
    k = max(entropy.RANDOMS_TO_MANAGE)
    register_random(r2)
    assert len(entropy.RANDOMS_TO_MANAGE) == n_registered + 2
    with deterministic_PRNG(0):
        del r1
        gc_collect()
        assert k not in entropy.RANDOMS_TO_MANAGE, 'r1 has been garbage-collected'
        assert len(entropy.RANDOMS_TO_MANAGE) == n_registered + 1
        r2.seed(4)
        register_random(r3)
        r3.seed(4)
        s4 = r3.getstate()
    assert r2.getstate() == s2, 'reset previously registered random state'
    assert r3.getstate() == s4, 'retained state when registered within the context'

@pytest.mark.skipif(PYPY, reason="We can't guard against bad no-reference patterns in pypy.")
def test_passing_unreferenced_instance_raises():
    if False:
        return 10
    with pytest.raises(ReferenceError):
        register_random(random.Random(0))

@pytest.mark.skipif(PYPY, reason="We can't guard against bad no-reference patterns in pypy.")
def test_passing_unreferenced_instance_within_function_scope_raises():
    if False:
        return 10

    def f():
        if False:
            print('Hello World!')
        register_random(random.Random(0))
    with pytest.raises(ReferenceError):
        f()

@pytest.mark.skipif(PYPY, reason="We can't guard against bad no-reference patterns in pypy.")
def test_passing_referenced_instance_within_function_scope_warns():
    if False:
        return 10

    def f():
        if False:
            print('Hello World!')
        r = random.Random(0)
        register_random(r)
    with pytest.warns(HypothesisWarning, match='It looks like `register_random` was passed an object that could be garbage collected'):
        f()

@pytest.mark.filterwarnings('ignore:It looks like `register_random` was passed an object that could be garbage collected')
@pytest.mark.skipif(PYPY, reason="We can't guard against bad no-reference patterns in pypy.")
def test_register_random_within_nested_function_scope():
    if False:
        for i in range(10):
            print('nop')
    n_registered = len(entropy.RANDOMS_TO_MANAGE)

    def f():
        if False:
            return 10
        r = random.Random()
        register_random(r)
        assert len(entropy.RANDOMS_TO_MANAGE) == n_registered + 1
    f()
    gc_collect()
    assert len(entropy.RANDOMS_TO_MANAGE) == n_registered