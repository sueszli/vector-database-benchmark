from ..utils import DummyObject, requires_backends

class TFBertTokenizer(metaclass=DummyObject):
    _backends = ['tensorflow_text']

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        requires_backends(self, ['tensorflow_text'])