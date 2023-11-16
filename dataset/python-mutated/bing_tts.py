from .tts import TTS, TTSValidator
from mycroft.configuration import Configuration

class BingTTS(TTS):

    def __init__(self, lang, config):
        if False:
            return 10
        super(BingTTS, self).__init__(lang, config, BingTTSValidator(self))
        self.type = 'wav'
        from bingtts import Translator
        self.config = Configuration.get().get('tts', {}).get('bing', {})
        api = self.config.get('api_key')
        self.bing = Translator(api)
        self.gender = self.config.get('gender', 'Male')
        self.format = self.config.get('format', 'riff-16khz-16bit-mono-pcm')

    def get_tts(self, sentence, wav_file):
        if False:
            print('Hello World!')
        output = self.bing.speak(sentence, self.lang, self.gender, self.format)
        with open(wav_file, 'w') as f:
            f.write(output)
        return (wav_file, None)

class BingTTSValidator(TTSValidator):

    def __init__(self, tts):
        if False:
            return 10
        super(BingTTSValidator, self).__init__(tts)

    def validate_dependencies(self):
        if False:
            while True:
                i = 10
        try:
            from bingtts import Translator
        except ImportError:
            raise Exception('BingTTS dependencies not installed, please run pip install git+https://github.com/westparkcom/Python-Bing-TTS.git ')

    def validate_lang(self):
        if False:
            i = 10
            return i + 15
        pass

    def validate_connection(self):
        if False:
            while True:
                i = 10
        pass

    def get_tts_class(self):
        if False:
            while True:
                i = 10
        return BingTTS