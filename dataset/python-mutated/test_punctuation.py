import unittest
from TTS.tts.utils.text.punctuation import _DEF_PUNCS, Punctuation

class PunctuationTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.punctuation = Punctuation()
        self.test_texts = [('This, is my text ... to be striped !! from text?', 'This is my text to be striped from text'), ('This, is my text ... to be striped !! from text', 'This is my text to be striped from text'), ('This, is my text ... to be striped  from text?', 'This is my text to be striped  from text'), ('This, is my text to be striped from text', 'This is my text to be striped from text')]

    def test_get_set_puncs(self):
        if False:
            print('Hello World!')
        self.punctuation.puncs = '-='
        self.assertEqual(self.punctuation.puncs, '-=')
        self.punctuation.puncs = _DEF_PUNCS
        self.assertEqual(self.punctuation.puncs, _DEF_PUNCS)

    def test_strip_punc(self):
        if False:
            return 10
        for (text, gt) in self.test_texts:
            text_striped = self.punctuation.strip(text)
            self.assertEqual(text_striped, gt)

    def test_strip_restore(self):
        if False:
            while True:
                i = 10
        for (text, gt) in self.test_texts:
            (text_striped, puncs_map) = self.punctuation.strip_to_restore(text)
            text_restored = self.punctuation.restore(text_striped, puncs_map)
            self.assertEqual(' '.join(text_striped), gt)
            self.assertEqual(text_restored[0], text)