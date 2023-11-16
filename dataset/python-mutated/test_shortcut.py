from collections import Counter
import nni
from nni.mutable._notimplemented import randint, lognormal, qlognormal

def test_choice():
    if False:
        return 10
    t = nni.choice('t', ['a', 'b', 'c'])
    assert repr(t) == "Categorical(['a', 'b', 'c'], label='t')"

def test_randint():
    if False:
        print('Hello World!')
    t = randint('x', 1, 5)
    assert repr(t) == "RandomInteger([1, 2, 3, 4], label='x')"

def test_uniform():
    if False:
        i = 10
        return i + 15
    t = nni.uniform('x', 0, 1)
    assert repr(t) == "Numerical(0, 1, label='x')"

def test_quniform():
    if False:
        i = 10
        return i + 15
    t = nni.quniform('x', 2.5, 5.5, 2.0)
    assert repr(t) == "Numerical(2.5, 5.5, q=2.0, label='x')"
    t = nni.quniform('x', 0.5, 3.5, 1).int()
    counter = Counter()
    for _ in range(900):
        counter[t.random()] += 1
    for (key, value) in counter.items():
        assert 250 <= value <= 350
        assert isinstance(key, int)
        assert key in [1, 2, 3]

def test_loguniform():
    if False:
        while True:
            i = 10
    t = nni.loguniform('x', 1e-05, 0.001)
    assert repr(t) == "Numerical(1e-05, 0.001, log_distributed=True, label='x')"
    for _ in range(100):
        assert 1e-05 < t.random() < 0.001

def test_qloguniform():
    if False:
        for i in range(10):
            print('nop')
    t = nni.qloguniform('x', 1e-05, 0.001, 0.0001)
    assert repr(t) == "Numerical(1e-05, 0.001, q=0.0001, log_distributed=True, label='x')"
    for x in t.grid(granularity=8):
        assert (x == 1e-05 or abs(x - round(x / 0.0001) * 0.0001) < 1e-12) and 1e-05 <= x <= 0.001

def test_normal():
    if False:
        while True:
            i = 10
    t = nni.normal('x', 0, 1)
    assert repr(t) == "Numerical(-inf, inf, mu=0, sigma=1, label='x')"
    assert -4 < t.random() < 4

def test_qnormal():
    if False:
        i = 10
        return i + 15
    t = nni.qnormal('x', 0.0, 1.0, 0.1)
    assert repr(t) == "Numerical(-inf, inf, mu=0.0, sigma=1.0, q=0.1, label='x')"

def test_lognormal():
    if False:
        print('Hello World!')
    t = lognormal('x', 4.0, 2.0)
    assert repr(t) == "Numerical(-inf, inf, mu=4.0, sigma=2.0, log_distributed=True, label='x')"
    assert 54 < list(t.grid(granularity=1))[0] < 55

def test_qlognormal():
    if False:
        i = 10
        return i + 15
    t = qlognormal('x', 4.0, 2.0, 1.0)
    assert repr(t) == "Numerical(-inf, inf, mu=4.0, sigma=2.0, q=1.0, log_distributed=True, label='x')"