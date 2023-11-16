from coalib.bearlib.aspects import Root, aspectclass, TasteError
from coalib.bearlib.aspects.base import aspectbase
import pytest

class SubAspectTest:

    def test_class(self, SubAspect):
        if False:
            print('Hello World!')
        assert isinstance(SubAspect, aspectclass)
        assert issubclass(SubAspect, aspectbase)

    def test_class_tastes_recreation(self, SubAspect):
        if False:
            print('Hello World!')
        assert SubAspect.tastes is not SubAspect.tastes

    def test_class_tastes_items(self, SubAspect, SubAspect_tastes):
        if False:
            while True:
                i = 10
        tastes = SubAspect.tastes
        for (name, taste) in SubAspect_tastes.items():
            assert taste is tastes.pop(name)
        assert not tastes

    def test__init__unavailable_taste(self, SubAspect, SubAspect_taste_values):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TasteError) as exc:
            SubAspect('cs', **SubAspect_taste_values)
        assert exc.match('is not available')

    def test_tastes(self, SubAspect, SubAspect_tastes):
        if False:
            while True:
                i = 10
        for language in ['py', 'cs']:
            aspect = SubAspect(language)
            taste_values = aspect.tastes
            for (name, taste) in SubAspect_tastes.items():
                if not taste.languages or language in taste.languages:
                    assert getattr(aspect, name) == taste_values[name] == taste.default
                else:
                    with pytest.raises(TasteError) as exc:
                        getattr(aspect, name)
                    assert exc.match('is not available')
                    assert name not in taste_values

    def test__eq__(self, RootAspect, SubAspect, SubAspect_taste_values):
        if False:
            print('Hello World!')
        assert SubAspect('py') == SubAspect('py')
        assert SubAspect('py', **SubAspect_taste_values) == SubAspect('py', **SubAspect_taste_values)
        assert not SubAspect('py') == RootAspect('py')
        assert not SubAspect('py') == Root('py')
        assert not SubAspect('py') == SubAspect('py', **SubAspect_taste_values)
        assert not SubAspect('py', **SubAspect_taste_values) == SubAspect('py')

    def test__ne__(self, RootAspect, SubAspect, SubAspect_taste_values):
        if False:
            i = 10
            return i + 15
        assert not SubAspect('py') != SubAspect('py')
        assert not SubAspect('py', **SubAspect_taste_values) != SubAspect('py', **SubAspect_taste_values)
        assert SubAspect('py') != RootAspect('py')
        assert SubAspect('py') != Root('py')
        assert SubAspect('py') != SubAspect('py', **SubAspect_taste_values)
        assert SubAspect('py', **SubAspect_taste_values) != SubAspect('py')