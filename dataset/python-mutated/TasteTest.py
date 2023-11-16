from coalib.bearlib.aspects import Taste
from coalib.bearlib.aspects.taste import TasteMeta
from coalib.bearlib.languages import Languages

class TasteTest:

    def test_class(self):
        if False:
            while True:
                i = 10
        assert isinstance(Taste, TasteMeta)

    def test_class__getitem__(self):
        if False:
            return 10
        typed = Taste[int]
        assert typed.cast_type is int
        assert typed.__name__ == typed.__qualname__ == 'Taste[int]'

    def test__init__defaults(self):
        if False:
            print('Hello World!')
        taste = Taste()
        assert taste.description is ''
        assert taste.suggested_values is ()
        assert taste.default is None
        assert type(taste.languages) is Languages
        assert not taste.languages
        assert not len(taste.languages)
        assert taste.languages == ()

    def test__get__(self, SubAspect, SubAspect_tastes, SubAspect_taste_values):
        if False:
            print('Hello World!')
        using_default_values = SubAspect('py')
        using_custom_values = SubAspect('py', **SubAspect_taste_values)
        for (name, taste) in SubAspect_tastes.items():
            assert getattr(SubAspect, name) is taste
            assert getattr(using_default_values, name) == taste.default
            assert getattr(using_custom_values, name) == SubAspect_taste_values[name]