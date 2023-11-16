import pytest
from astropy.io.misc import fnpickle, fnunpickle
from astropy.utils.exceptions import AstropyDeprecationWarning

def test_fnpickling_simple(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    "\n    Tests the `fnpickle` and `fnupickle` functions' basic operation by\n    pickling and unpickling a string, using both a filename and a\n    file.\n    "
    fn = str(tmp_path / 'test1.pickle')
    obj1 = 'astring'
    with pytest.warns(AstropyDeprecationWarning, match='Use pickle from standard library'):
        fnpickle(obj1, fn)
        res = fnunpickle(fn, 0)
        assert obj1 == res
        with open(fn, 'wb') as f:
            fnpickle(obj1, f)
        with open(fn, 'rb') as f:
            res = fnunpickle(f)
            assert obj1 == res

class ToBePickled:

    def __init__(self, item):
        if False:
            for i in range(10):
                print('nop')
        self.item = item

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, ToBePickled):
            return self.item == other.item
        else:
            return False

def test_fnpickling_class(tmp_path):
    if False:
        i = 10
        return i + 15
    "\n    Tests the `fnpickle` and `fnupickle` functions' ability to pickle\n    and unpickle custom classes.\n    "
    fn = str(tmp_path / 'test2.pickle')
    obj1 = 'astring'
    obj2 = ToBePickled(obj1)
    with pytest.warns(AstropyDeprecationWarning, match='Use pickle from standard library'):
        fnpickle(obj2, fn)
        res = fnunpickle(fn)
    assert res == obj2

def test_fnpickling_protocol(tmp_path):
    if False:
        return 10
    "\n    Tests the `fnpickle` and `fnupickle` functions' ability to pickle\n    and unpickle pickle files from all protocols.\n    "
    import pickle
    obj1 = 'astring'
    obj2 = ToBePickled(obj1)
    for p in range(pickle.HIGHEST_PROTOCOL + 1):
        fn = str(tmp_path / f'testp{p}.pickle')
        with pytest.warns(AstropyDeprecationWarning, match='Use pickle from standard library'):
            fnpickle(obj2, fn, protocol=p)
            res = fnunpickle(fn)
        assert res == obj2

def test_fnpickling_many(tmp_path):
    if False:
        return 10
    "\n    Tests the `fnpickle` and `fnupickle` functions' ability to pickle\n    and unpickle multiple objects from a single file.\n    "
    fn = str(tmp_path / 'test3.pickle')
    obj3 = 328.3432
    obj4 = 'blahblahfoo'
    with pytest.warns(AstropyDeprecationWarning, match='Use pickle from standard library'):
        fnpickle(obj3, fn)
        fnpickle(obj4, fn, append=True)
        res = fnunpickle(fn, number=-1)
        assert len(res) == 2
        assert res[0] == obj3
        assert res[1] == obj4
        fnpickle(obj4, fn, append=True)
        res = fnunpickle(fn, number=2)
        assert len(res) == 2
        with pytest.raises(EOFError):
            fnunpickle(fn, number=5)