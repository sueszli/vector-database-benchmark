"""
Transcription module tests
"""
import unittest
import numpy as np
import soundfile as sf
from scipy import signal
from txtai.pipeline import Transcription
from utils import Utils

class TestTranscription(unittest.TestCase):
    """
    Transcription tests.
    """

    def testArray(self):
        if False:
            print('Hello World!')
        '\n        Test audio data to text transcription\n        '
        transcribe = Transcription()
        (raw, samplerate) = sf.read(Utils.PATH + '/Make_huge_profits.wav')
        self.assertEqual(transcribe((raw, samplerate)), 'Make huge profits without working make up to one hundred thousand dollars a day')
        self.assertEqual(transcribe(raw, samplerate), 'Make huge profits without working make up to one hundred thousand dollars a day')

    def testChunks(self):
        if False:
            print('Hello World!')
        '\n        Test splitting transcription into chunks\n        '
        transcribe = Transcription()
        result = transcribe(Utils.PATH + '/Make_huge_profits.wav', join=False)[0]
        self.assertIsInstance(result['raw'], np.ndarray)
        self.assertIsNotNone(result['rate'])
        self.assertEqual(result['text'], 'Make huge profits without working make up to one hundred thousand dollars a day')

    def testFile(self):
        if False:
            while True:
                i = 10
        '\n        Test audio file to text transcription\n        '
        transcribe = Transcription()
        self.assertEqual(transcribe(Utils.PATH + '/Make_huge_profits.wav'), 'Make huge profits without working make up to one hundred thousand dollars a day')

    def testResample(self):
        if False:
            print('Hello World!')
        '\n        Test resampled audio file to text transcription\n        '
        transcribe = Transcription()
        (raw, samplerate) = sf.read(Utils.PATH + '/Make_huge_profits.wav')
        samples = round(len(raw) * float(22050) / samplerate)
        (raw, samplerate) = (signal.resample(raw, samples), 22050)
        self.assertEqual(transcribe(raw, samplerate), 'Make huge profits without working make up to one hundred thousand dollars a day')

    def testStereo(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test audio file in stereo to text transcription\n        '
        transcribe = Transcription()
        (raw, samplerate) = sf.read(Utils.PATH + '/Make_huge_profits.wav')
        raw = np.column_stack((raw, raw))
        self.assertEqual(transcribe(raw, samplerate), 'Make huge profits without working make up to one hundred thousand dollars a day')