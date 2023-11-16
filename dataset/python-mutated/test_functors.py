import functools
import dill
dill.settings['recurse'] = True

def f(a, b, c):
    if False:
        i = 10
        return i + 15
    pass

def g(a, b, c=2):
    if False:
        print('Hello World!')
    pass

def h(a=1, b=2, c=3):
    if False:
        while True:
            i = 10
    pass

def test_functools():
    if False:
        i = 10
        return i + 15
    fp = functools.partial(f, 1, 2)
    gp = functools.partial(g, 1, c=2)
    hp = functools.partial(h, 1, c=2)
    bp = functools.partial(int, base=2)
    assert dill.pickles(fp, safe=True)
    assert dill.pickles(gp, safe=True)
    assert dill.pickles(hp, safe=True)
    assert dill.pickles(bp, safe=True)
if __name__ == '__main__':
    test_functools()