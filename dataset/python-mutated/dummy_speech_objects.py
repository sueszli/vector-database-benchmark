from ..utils import DummyObject, requires_backends

class ASTFeatureExtractor(metaclass=DummyObject):
    _backends = ['speech']

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        requires_backends(self, ['speech'])

class Speech2TextFeatureExtractor(metaclass=DummyObject):
    _backends = ['speech']

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        requires_backends(self, ['speech'])