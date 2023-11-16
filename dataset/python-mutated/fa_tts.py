import requests
from .tts import TTSValidator
from .remote_tts import RemoteTTS

class FATTS(RemoteTTS):
    PARAMS = {'voice[name]': 'cmu-slt-hsmm', 'input[type]': 'TEXT', 'input[locale]': 'en_US', 'input[content]': 'Hello World', 'output[format]': 'WAVE_FILE', 'output[type]': 'AUDIO'}

    def __init__(self, lang, config):
        if False:
            i = 10
            return i + 15
        super(FATTS, self).__init__(lang, config, '/say', FATTSValidator(self))

    def build_request_params(self, sentence):
        if False:
            i = 10
            return i + 15
        params = self.PARAMS.copy()
        params['voice[name]'] = self.voice
        params['input[locale]'] = self.lang
        params['input[content]'] = sentence.encode('utf-8')
        return params

class FATTSValidator(TTSValidator):

    def __init__(self, tts):
        if False:
            print('Hello World!')
        super(FATTSValidator, self).__init__(tts)

    def validate_lang(self):
        if False:
            i = 10
            return i + 15
        pass

    def validate_connection(self):
        if False:
            print('Hello World!')
        try:
            resp = requests.get(self.tts.url + '/info/version', verify=False)
            content = resp.json()
            if content.get('product', '').find('FA-TTS') < 0:
                raise Exception('Invalid FA-TTS server.')
        except Exception:
            raise Exception('FA-TTS server could not be verified. Check your connection to the server: ' + self.tts.url)

    def get_tts_class(self):
        if False:
            return 10
        return FATTS