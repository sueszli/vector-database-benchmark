import unittest
from mycroft.client.speech.hotword_factory import HotWordFactory

class PocketSphinxTest(unittest.TestCase):

    def testDefault(self):
        if False:
            i = 10
            return i + 15
        config = {'hey mycroft': {'module': 'pocketsphinx', 'phonemes': 'HH EY . M AY K R AO F T', 'threshold': 1e-90}}
        p = HotWordFactory.create_hotword('hey mycroft', config)
        config = config['hey mycroft']
        self.assertEqual(config['phonemes'], p.phonemes)
        self.assertEqual(config['threshold'], p.threshold)

    def testInvalid(self):
        if False:
            print('Hello World!')
        config = {'hey Zeds': {'module': 'pocketsphinx', 'phonemes': 'ZZZZZZZZZ', 'threshold': 1e-90}}
        p = HotWordFactory.create_hotword('hey Zeds', config)
        self.assertEqual(p.phonemes, 'HH EY . M AY K R AO F T')
        self.assertEqual(p.key_phrase, 'hey mycroft')

    def testVictoria(self):
        if False:
            return 10
        config = {'hey victoria': {'module': 'pocketsphinx', 'phonemes': 'HH EY . V IH K T AO R IY AH', 'threshold': 1e-90}}
        p = HotWordFactory.create_hotword('hey victoria', config)
        config = config['hey victoria']
        self.assertEqual(config['phonemes'], p.phonemes)
        self.assertEqual(p.key_phrase, 'hey victoria')