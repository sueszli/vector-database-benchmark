from celery.app.annotations import MapAnnotation, prepare
from celery.utils.imports import qualname

class MyAnnotation:
    foo = 65

class AnnotationCase:

    def setup_method(self):
        if False:
            return 10

        @self.app.task(shared=False)
        def add(x, y):
            if False:
                i = 10
                return i + 15
            return x + y
        self.add = add

        @self.app.task(shared=False)
        def mul(x, y):
            if False:
                return 10
            return x * y
        self.mul = mul

class test_MapAnnotation(AnnotationCase):

    def test_annotate(self):
        if False:
            while True:
                i = 10
        x = MapAnnotation({self.add.name: {'foo': 1}})
        assert x.annotate(self.add) == {'foo': 1}
        assert x.annotate(self.mul) is None

    def test_annotate_any(self):
        if False:
            while True:
                i = 10
        x = MapAnnotation({'*': {'foo': 2}})
        assert x.annotate_any() == {'foo': 2}
        x = MapAnnotation()
        assert x.annotate_any() is None

class test_prepare(AnnotationCase):

    def test_dict_to_MapAnnotation(self):
        if False:
            for i in range(10):
                print('nop')
        x = prepare({self.add.name: {'foo': 3}})
        assert isinstance(x[0], MapAnnotation)

    def test_returns_list(self):
        if False:
            i = 10
            return i + 15
        assert prepare(1) == [1]
        assert prepare([1]) == [1]
        assert prepare((1,)) == [1]
        assert prepare(None) == ()

    def test_evalutes_qualnames(self):
        if False:
            for i in range(10):
                print('nop')
        assert prepare(qualname(MyAnnotation))[0]().foo == 65
        assert prepare([qualname(MyAnnotation)])[0]().foo == 65