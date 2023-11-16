"""Entropy coder model registrar."""

class ModelFactory(object):
    """Factory of encoder/decoder models."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._model_dictionary = dict()

    def RegisterModel(self, entropy_coder_model_name, entropy_coder_model_factory):
        if False:
            while True:
                i = 10
        self._model_dictionary[entropy_coder_model_name] = entropy_coder_model_factory

    def CreateModel(self, model_name):
        if False:
            print('Hello World!')
        current_model_factory = self._model_dictionary[model_name]
        return current_model_factory()

    def GetAvailableModels(self):
        if False:
            while True:
                i = 10
        return self._model_dictionary.keys()
_model_registry = ModelFactory()

def GetModelRegistry():
    if False:
        while True:
            i = 10
    return _model_registry

class RegisterEntropyCoderModel(object):

    def __init__(self, model_name):
        if False:
            for i in range(10):
                print('nop')
        self._model_name = model_name

    def __call__(self, f):
        if False:
            while True:
                i = 10
        _model_registry.RegisterModel(self._model_name, f)
        return f