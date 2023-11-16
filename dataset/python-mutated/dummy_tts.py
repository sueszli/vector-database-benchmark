"""A Dummy TTS without any audio output."""
from mycroft.util.log import LOG
from .tts import TTS, TTSValidator

class DummyTTS(TTS):

    def __init__(self, lang, config):
        if False:
            while True:
                i = 10
        super().__init__(lang, config, DummyValidator(self), 'wav')

    def execute(self, sentence, ident=None, listen=False):
        if False:
            print('Hello World!')
        "Don't do anything, return nothing."
        LOG.info('Mycroft: {}'.format(sentence))
        self.end_audio(listen)
        return None

class DummyValidator(TTSValidator):
    """Do no tests."""

    def __init__(self, tts):
        if False:
            return 10
        super().__init__(tts)

    def validate_lang(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def validate_connection(self):
        if False:
            i = 10
            return i + 15
        pass

    def get_tts_class(self):
        if False:
            for i in range(10):
                print('nop')
        return DummyTTS