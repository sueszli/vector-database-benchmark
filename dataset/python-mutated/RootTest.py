from coalib.bearlib.aspects import Root, aspectclass
from coalib.bearlib.aspects.base import aspectbase

class RootTest:

    def test_class(self):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(Root, aspectclass)
        assert issubclass(Root, aspectbase)

    def test_class_subaspects(self):
        if False:
            i = 10
            return i + 15
        assert isinstance(Root.subaspects, dict)

    def test_class_parent(self):
        if False:
            for i in range(10):
                print('nop')
        assert Root.parent is None

    def test_class_tastes(self):
        if False:
            while True:
                i = 10
        assert Root.tastes == {}

    def test__eq__(self, RootAspect):
        if False:
            for i in range(10):
                print('nop')
        assert Root('py') == Root('py')
        assert not Root('py') == RootAspect('py')

    def test__ne__(self, RootAspect):
        if False:
            while True:
                i = 10
        assert not Root('py') != Root('py')
        assert Root('py') != RootAspect('py')