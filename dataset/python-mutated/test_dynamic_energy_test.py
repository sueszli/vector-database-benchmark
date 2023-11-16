import audioop
from unittest import TestCase, mock
from speech_recognition import AudioSource
from mycroft.client.speech.mic import ResponsiveRecognizer

class MockStream:

    def __init__(self):
        if False:
            return 10
        self.chunks = []

    def inject(self, chunk):
        if False:
            while True:
                i = 10
        self.chunks.append(chunk)

    def read(self, chunk_size):
        if False:
            for i in range(10):
                print('nop')
        result = self.chunks[0]
        if len(self.chunks) > 1:
            self.chunks = self.chunks[1:]
        return result

class MockSource(AudioSource):

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            print('Hello World!')
        pass

    def __init__(self, stream=None):
        if False:
            while True:
                i = 10
        self.stream = stream if stream else MockStream()
        self.CHUNK = 1024
        self.SAMPLE_RATE = 16000
        self.SAMPLE_WIDTH = 2

class MockHotwordEngine(mock.Mock):

    def __init__(self, *arg, **kwarg):
        if False:
            return 10
        super().__init__(*arg, **kwarg)
        self.num_phonemes = 10

class DynamicEnergytest(TestCase):

    def testMaxAudioWithBaselineShift(self):
        if False:
            i = 10
            return i + 15
        low_base = b'\x10\x00\x01\x00' * 100
        higher_base = b'\x01\x00\x00\x01' * 100
        source = MockSource()
        for i in range(100):
            source.stream.inject(low_base)
        source.stream.inject(higher_base)
        recognizer = ResponsiveRecognizer(MockHotwordEngine())
        sec_per_buffer = float(source.CHUNK) / (source.SAMPLE_RATE * source.SAMPLE_WIDTH)
        test_seconds = 30.0
        while test_seconds > 0:
            test_seconds -= sec_per_buffer
            data = source.stream.read(source.CHUNK)
            energy = recognizer.calc_energy(data, source.SAMPLE_WIDTH)
            recognizer._adjust_threshold(energy, sec_per_buffer)
        higher_base_energy = audioop.rms(higher_base, source.SAMPLE_WIDTH)
        delta_below_threshold = recognizer.energy_threshold - higher_base_energy
        min_delta = higher_base_energy * 0.5
        assert abs(delta_below_threshold - min_delta) < 1