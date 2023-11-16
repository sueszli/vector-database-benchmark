from sympy.core.sorting import default_sort_key, ordered
from sympy.testing.pytest import raises
from sympy.abc import x

def test_default_sort_key():
    if False:
        for i in range(10):
            print('nop')
    func = lambda x: x
    assert sorted([func, x, func], key=default_sort_key) == [func, func, x]

    class C:

        def __repr__(self):
            if False:
                print('Hello World!')
            return 'x.y'
    func = C()
    assert sorted([x, func], key=default_sort_key) == [func, x]

def test_ordered():
    if False:
        i = 10
        return i + 15
    assert list(ordered([{1: 3, 2: 4, 9: 10}, {1: 3}])) == [{1: 3}, {1: 3, 2: 4, 9: 10}]
    l = [1, 1]
    assert list(ordered(l, warn=True)) == l
    l = [[1], [2], [1]]
    assert list(ordered(l, warn=True)) == [[1], [1], [2]]
    raises(ValueError, lambda : list(ordered(['a', 'ab'], keys=[lambda x: x[0]], default=False, warn=True)))