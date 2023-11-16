from ..utils import DummyObject, requires_backends

class Pop2PianoFeatureExtractor(metaclass=DummyObject):
    _backends = ['essentia', 'librosa', 'pretty_midi', 'scipy', 'torch']

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        requires_backends(self, ['essentia', 'librosa', 'pretty_midi', 'scipy', 'torch'])

class Pop2PianoTokenizer(metaclass=DummyObject):
    _backends = ['essentia', 'librosa', 'pretty_midi', 'scipy', 'torch']

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        requires_backends(self, ['essentia', 'librosa', 'pretty_midi', 'scipy', 'torch'])

class Pop2PianoProcessor(metaclass=DummyObject):
    _backends = ['essentia', 'librosa', 'pretty_midi', 'scipy', 'torch']

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        requires_backends(self, ['essentia', 'librosa', 'pretty_midi', 'scipy', 'torch'])