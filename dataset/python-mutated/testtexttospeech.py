"""
TextToSpeech module tests
"""
import unittest
from unittest.mock import patch
from txtai.pipeline import TextToSpeech

class TestTextToSpeech(unittest.TestCase):
    """
    TextToSpeech tests.
    """

    def testTextToSpeech(self):
        if False:
            return 10
        '\n        Test generating speech for text\n        '
        tts = TextToSpeech()
        self.assertGreater(len(tts('This is a test')), 0)

    @patch('onnxruntime.get_available_providers')
    @patch('torch.cuda.is_available')
    def testProviders(self, cuda, providers):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that GPU provider is detected\n        '
        cuda.return_value = True
        providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        tts = TextToSpeech()
        self.assertEqual(tts.providers()[0][0], 'CUDAExecutionProvider')