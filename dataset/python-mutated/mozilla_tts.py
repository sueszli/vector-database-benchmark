import requests
from .tts import TTS, TTSValidator
from mycroft.configuration import Configuration

class MozillaTTS(TTS):

    def __init__(self, lang='en-us', config=None):
        if False:
            while True:
                i = 10
        if config is None:
            self.config = Configuration.get().get('tts', {}).get('mozilla', {})
        else:
            self.config = config
        super(MozillaTTS, self).__init__(lang, self.config, MozillaTTSValidator(self))
        self.url = self.config['url'] + '/api/tts'
        self.type = 'wav'

    def get_tts(self, sentence, wav_file):
        if False:
            return 10
        response = requests.get(self.url, params={'text': sentence})
        with open(wav_file, 'wb') as f:
            f.write(response.content)
        return (wav_file, None)

class MozillaTTSValidator(TTSValidator):

    def __init__(self, tts):
        if False:
            while True:
                i = 10
        super(MozillaTTSValidator, self).__init__(tts)

    def validate_dependencies(self):
        if False:
            i = 10
            return i + 15
        pass

    def validate_lang(self):
        if False:
            i = 10
            return i + 15
        pass

    def validate_connection(self):
        if False:
            while True:
                i = 10
        url = self.tts.config['url']
        response = requests.get(url)
        if not response.status_code == 200:
            raise ConnectionRefusedError

    def get_tts_class(self):
        if False:
            return 10
        return MozillaTTS