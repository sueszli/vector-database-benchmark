import requests
from .tts import TTSValidator
from .remote_tts import RemoteTTS

class MaryTTS(RemoteTTS):
    PARAMS = {'LOCALE': 'en_US', 'VOICE': 'cmu-slt-hsmm', 'INPUT_TEXT': 'Hello World', 'INPUT_TYPE': 'TEXT', 'AUDIO': 'WAVE_FILE', 'OUTPUT_TYPE': 'AUDIO'}

    def __init__(self, lang, config):
        if False:
            while True:
                i = 10
        super(MaryTTS, self).__init__(lang, config, config.get('url'), '/process', MaryTTSValidator(self))

    def build_request_params(self, sentence):
        if False:
            return 10
        params = self.PARAMS.copy()
        params['LOCALE'] = self.lang
        params['VOICE'] = self.voice
        params['INPUT_TEXT'] = sentence.encode('utf-8')
        return params

class MaryTTSValidator(TTSValidator):

    def __init__(self, tts):
        if False:
            return 10
        super(MaryTTSValidator, self).__init__(tts)

    def validate_lang(self):
        if False:
            i = 10
            return i + 15
        pass

    def validate_connection(self):
        if False:
            return 10
        try:
            resp = requests.get(self.tts.url + '/version', verify=False)
            if resp.status_code == 200:
                return True
        except Exception:
            raise Exception('MaryTTS server could not be verified. Check your connection to the server: ' + self.tts.url)

    def get_tts_class(self):
        if False:
            while True:
                i = 10
        return MaryTTS