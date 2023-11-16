import unittest
from unittest.mock import patch
import os
from speech_recognition import WavFile
from mycroft.client.speech.listener import RecognizerLoop
from mycroft.configuration import Configuration
from test.util import base_config
DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

class PocketSphinxRecognizerTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        with patch('mycroft.configuration.Configuration.get') as mock_config_get:
            conf = base_config()
            conf['hotwords']['hey mycroft']['module'] = 'pocketsphinx'
            mock_config_get.return_value = conf
            rl = RecognizerLoop()
            self.recognizer = RecognizerLoop.create_wake_word_recognizer(rl)

    def testRecognizerWrapper(self):
        if False:
            for i in range(10):
                print('nop')
        source = WavFile(os.path.join(DATA_DIR, 'hey_mycroft.wav'))
        with source as audio:
            assert self.recognizer.found_wake_word(audio.stream.read())
        source = WavFile(os.path.join(DATA_DIR, 'mycroft.wav'))
        with source as audio:
            assert self.recognizer.found_wake_word(audio.stream.read())

    def testRecognitionInLongerUtterance(self):
        if False:
            for i in range(10):
                print('nop')
        source = WavFile(os.path.join(DATA_DIR, 'weather_mycroft.wav'))
        with source as audio:
            assert self.recognizer.found_wake_word(audio.stream.read())

    @patch.object(Configuration, 'get')
    def testRecognitionFallback(self, mock_config_get):
        if False:
            return 10
        "If language config doesn't exist set default (english)"
        conf = base_config()
        conf['hotwords']['hey mycroft'] = {'lang': 'DOES NOT EXIST', 'module': 'pocketsphinx', 'phonemes': 'HH EY . M AY K R AO F T', 'threshold': 1e-90}
        conf['lang'] = 'DOES NOT EXIST'
        mock_config_get.return_value = conf
        rl = RecognizerLoop()
        ps_hotword = RecognizerLoop.create_wake_word_recognizer(rl)
        expected = 'en-us'
        res = ps_hotword.decoder.get_config().get_string('-hmm')
        self.assertEqual(expected, res.split('/')[-2])
        self.assertEqual('does not exist', ps_hotword.lang)

class LocalRecognizerInitTest(unittest.TestCase):

    @patch.object(Configuration, 'get')
    def testListenerConfig(self, mock_config_get):
        if False:
            return 10
        'Ensure that the fallback method collecting phonemes etc.\n        from the listener config works.\n        '
        test_config = base_config()
        mock_config_get.return_value = test_config
        rl = RecognizerLoop()
        self.assertEqual(rl.wakeword_recognizer.key_phrase, 'hey mycroft')
        test_config['listener']['wake_word'] = 'hey victoria'
        test_config['listener']['phonemes'] = 'HH EY . V IH K T AO R IY AH'
        test_config['listener']['threshold'] = 1e-90
        rl = RecognizerLoop()
        self.assertEqual(rl.wakeword_recognizer.key_phrase, 'hey victoria')
        test_config['listener']['wake_word'] = 'hey victoria'
        test_config['listener']['phonemes'] = 'ZZZZZZZZZZZZ'
        rl = RecognizerLoop()
        self.assertEqual(rl.wakeword_recognizer.key_phrase, 'hey mycroft')

    @patch.object(Configuration, 'get')
    def testHotwordConfig(self, mock_config_get):
        if False:
            i = 10
            return i + 15
        'Ensure that the fallback method collecting phonemes etc.\n        from the listener config works.\n        '
        test_config = base_config()
        mock_config_get.return_value = test_config
        test_config['listener']['phonemes'] = 'HH EY . V IH K T AO R IY AH'
        test_config['listener']['threshold'] = 1e-90
        steve_conf = {'model': 'pocketsphinx', 'phonemes': 'S T IY V .', 'threshold': 1e-42}
        test_config['hotwords']['steve'] = steve_conf
        test_config['listener']['wake_word'] = 'steve'
        rl = RecognizerLoop()
        self.assertEqual(rl.wakeword_recognizer.key_phrase, 'steve')
        test_config['listener']['phonemes'] = 'S T IY V .'
        test_config['listener']['threshold'] = 1e-90
        steve_conf = {'model': 'pocketsphinx'}
        test_config['hotwords']['steve'] = steve_conf
        test_config['listener']['wake_word'] = 'steve'
        rl = RecognizerLoop()
        self.assertEqual(rl.wakeword_recognizer.key_phrase, 'steve')
        self.assertEqual(rl.wakeword_recognizer.phonemes, 'S T IY V .')