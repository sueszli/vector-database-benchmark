from ..utils import DummyObject, requires_backends

class TFGPT2Tokenizer(metaclass=DummyObject):
    _backends = ['keras_nlp']

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        requires_backends(self, ['keras_nlp'])