from dlib import find_max_global, find_min_global
import dlib
from pytest import raises
from math import sin, cos, pi, exp, sqrt, pow

def test_global_optimization_nargs():
    if False:
        while True:
            i = 10
    w0 = find_max_global(lambda *args: sum(args), [0, 0, 0], [1, 1, 1], 10)
    w1 = find_min_global(lambda *args: sum(args), [0, 0, 0], [1, 1, 1], 10)
    assert w0 == ([1, 1, 1], 3)
    assert w1 == ([0, 0, 0], 0)
    w2 = find_max_global(lambda a, b, c, *args: a + b + c - sum(args), [0, 0, 0], [1, 1, 1], 10)
    w3 = find_min_global(lambda a, b, c, *args: a + b + c - sum(args), [0, 0, 0], [1, 1, 1], 10)
    assert w2 == ([1, 1, 1], 3)
    assert w3 == ([0, 0, 0], 0)
    with raises(Exception):
        find_max_global(lambda a, b: 0, [0, 0, 0], [1, 1, 1], 10)
    with raises(Exception):
        find_min_global(lambda a, b: 0, [0, 0, 0], [1, 1, 1], 10)
    with raises(Exception):
        find_max_global(lambda a, b, c, d, *args: 0, [0, 0, 0], [1, 1, 1], 10)
    with raises(Exception):
        find_min_global(lambda a, b, c, d, *args: 0, [0, 0, 0], [1, 1, 1], 10)

def F(a, b):
    if False:
        print('Hello World!')
    return -pow(a - 2, 2.0) - pow(b - 4, 2.0)

def G(x):
    if False:
        for i in range(10):
            print('nop')
    return 2 - pow(x - 5, 2.0)

def test_global_function_search():
    if False:
        print('Hello World!')
    spec_F = dlib.function_spec([-10, -10], [10, 10])
    spec_G = dlib.function_spec([-2], [6])
    opt = dlib.global_function_search([spec_F, spec_G])
    for i in range(15):
        next = opt.get_next_x()
        if next.function_idx == 0:
            a = next.x[0]
            b = next.x[1]
            next.set(F(a, b))
        else:
            x = next.x[0]
            next.set(G(x))
    [x, y, function_idx] = opt.get_best_function_eval()
    assert abs(y - 2) < 1e-07
    assert abs(x[0] - 5) < 1e-07
    assert function_idx == 1

def holder_table(x0, x1):
    if False:
        for i in range(10):
            print('nop')
    return -abs(sin(x0) * cos(x1) * exp(abs(1 - sqrt(x0 * x0 + x1 * x1) / pi)))

def test_on_holder_table():
    if False:
        while True:
            i = 10
    (x, y) = find_min_global(holder_table, [-10, -10], [10, 10], 200)
    assert y - -19.2085025679 < 1e-07