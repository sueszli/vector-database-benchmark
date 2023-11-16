from IPython.utils.dir2 import dir2
import pytest

class Base(object):
    x = 1
    z = 23

def test_base():
    if False:
        print('Hello World!')
    res = dir2(Base())
    assert 'x' in res
    assert 'z' in res
    assert 'y' not in res
    assert '__class__' in res
    assert res.count('x') == 1
    assert res.count('__class__') == 1

def test_SubClass():
    if False:
        while True:
            i = 10

    class SubClass(Base):
        y = 2
    res = dir2(SubClass())
    assert 'y' in res
    assert res.count('y') == 1
    assert res.count('x') == 1

def test_SubClass_with_trait_names_attr():
    if False:
        print('Hello World!')

    class SubClass(Base):
        y = 2
        trait_names = 44
    res = dir2(SubClass())
    assert 'trait_names' in res

def test_misbehaving_object_without_trait_names():
    if False:
        i = 10
        return i + 15

    class MisbehavingGetattr:

        def __getattr__(self, attr):
            if False:
                return 10
            raise KeyError('I should be caught')

        def some_method(self):
            if False:
                i = 10
                return i + 15
            return True

    class SillierWithDir(MisbehavingGetattr):

        def __dir__(self):
            if False:
                print('Hello World!')
            return ['some_method']
    for bad_klass in (MisbehavingGetattr, SillierWithDir):
        obj = bad_klass()
        assert obj.some_method()
        with pytest.raises(KeyError):
            obj.other_method()
        res = dir2(obj)
        assert 'some_method' in res