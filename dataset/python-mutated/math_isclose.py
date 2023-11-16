try:
    from math import isclose
except ImportError:
    print('SKIP')
    raise SystemExit

def test(a, b, **kwargs):
    if False:
        i = 10
        return i + 15
    print(isclose(a, b, **kwargs))

def test_combinations(a, b, **kwargs):
    if False:
        i = 10
        return i + 15
    test(a, a, **kwargs)
    test(a, b, **kwargs)
    test(b, a, **kwargs)
    test(b, b, **kwargs)
test_combinations(float('nan'), 1)
test_combinations(float('inf'), 1)
test_combinations(float('-inf'), 1)
test(1.0, 1.0, rel_tol=0.0, abs_tol=0.0)
test(2.35e-100, 2.35e-100, rel_tol=0.0, abs_tol=0.0)
test(2.1234e+100, 2.1234e+100, rel_tol=0.0, abs_tol=0.0)
test(1000.0, 1001.0, rel_tol=0.001)
test(1000.0, 1001.0, rel_tol=0.0001)
test(1000, 1001, rel_tol=0.001)
test(1000, 1001, rel_tol=0.0001)
test_combinations(0, 1, rel_tol=1.0)
test(0.0, 1e-10, abs_tol=1e-10, rel_tol=0.1)
test(0.0, 1e-10, abs_tol=0.0, rel_tol=0.1)
try:
    isclose(0, 0, abs_tol=-1)
except ValueError:
    print('ValueError')
try:
    isclose(0, 0, rel_tol=-1)
except ValueError:
    print('ValueError')