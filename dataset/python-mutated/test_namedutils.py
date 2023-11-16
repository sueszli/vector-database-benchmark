try:
    from cPickle import loads, dumps
except:
    from pickle import loads, dumps
from boltons.namedutils import namedlist, namedtuple
Point = namedtuple('Point', 'x, y', rename=True)
MutablePoint = namedlist('MutablePoint', 'x, y', rename=True)

def test_namedlist():
    if False:
        i = 10
        return i + 15
    p = MutablePoint(x=10, y=20)
    assert p == [10, 20]
    p[0] = 11
    assert p == [11, 20]
    p.x = 12
    assert p == [12, 20]

def test_namedlist_pickle():
    if False:
        print('Hello World!')
    p = MutablePoint(x=10, y=20)
    assert p == loads(dumps(p))

def test_namedtuple_pickle():
    if False:
        for i in range(10):
            print('nop')
    p = Point(x=10, y=20)
    assert p == loads(dumps(p))