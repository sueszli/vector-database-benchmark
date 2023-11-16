import pytest
from networkx.utils import powerlaw_sequence, random_weighted_sample, weighted_choice, zipf_rv

def test_degree_sequences():
    if False:
        return 10
    seq = powerlaw_sequence(10, seed=1)
    seq = powerlaw_sequence(10)
    assert len(seq) == 10

def test_zipf_rv():
    if False:
        i = 10
        return i + 15
    r = zipf_rv(2.3, xmin=2, seed=1)
    r = zipf_rv(2.3, 2, 1)
    r = zipf_rv(2.3)
    assert type(r), int
    pytest.raises(ValueError, zipf_rv, 0.5)
    pytest.raises(ValueError, zipf_rv, 2, xmin=0)

def test_random_weighted_sample():
    if False:
        while True:
            i = 10
    mapping = {'a': 10, 'b': 20}
    s = random_weighted_sample(mapping, 2, seed=1)
    s = random_weighted_sample(mapping, 2)
    assert sorted(s) == sorted(mapping.keys())
    pytest.raises(ValueError, random_weighted_sample, mapping, 3)

def test_random_weighted_choice():
    if False:
        for i in range(10):
            print('nop')
    mapping = {'a': 10, 'b': 0}
    c = weighted_choice(mapping, seed=1)
    c = weighted_choice(mapping)
    assert c == 'a'