from ..utils import DummyObject, requires_backends

class Pop2PianoFeatureExtractor(metaclass=DummyObject):
    _backends = ['music']

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        requires_backends(self, ['music'])

class Pop2PianoTokenizer(metaclass=DummyObject):
    _backends = ['music']

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        requires_backends(self, ['music'])