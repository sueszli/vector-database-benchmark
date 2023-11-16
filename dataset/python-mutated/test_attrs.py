from sqlalchemy.orm.exc import DetachedInstanceError
from warehouse.utils.attrs import make_repr

class TestMakeRepr:

    def test_on_class(self):
        if False:
            print('Hello World!')

        class Fake:
            foo = 'bar'
            __repr__ = make_repr('foo')
        assert repr(Fake()) == 'Fake(foo={})'.format(repr('bar'))

    def test_with_function(self):
        if False:
            for i in range(10):
                print('nop')

        class Fake:
            foo = 'bar'

            def __repr__(self):
                if False:
                    return 10
                self.__repr__ = make_repr('foo', _self=self)
                return self.__repr__()
        assert repr(Fake()) == 'Fake(foo={})'.format(repr('bar'))

    def test_with_raise(self):
        if False:
            for i in range(10):
                print('nop')

        class Fake:
            __repr__ = make_repr('foo')

            @property
            def foo(self):
                if False:
                    while True:
                        i = 10
                raise DetachedInstanceError
        assert repr(Fake()) == 'Fake(<detached>)'