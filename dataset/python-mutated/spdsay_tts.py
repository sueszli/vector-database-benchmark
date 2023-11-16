import subprocess
from .tts import TTS, TTSValidator

class SpdSay(TTS):

    def __init__(self, lang, config):
        if False:
            i = 10
            return i + 15
        super(SpdSay, self).__init__(lang, config, SpdSayValidator(self))

    def execute(self, sentence, ident=None, listen=False):
        if False:
            i = 10
            return i + 15
        self.begin_audio()
        subprocess.call(['spd-say', '-l', self.lang, '-t', self.voice, sentence])
        self.end_audio(listen)

class SpdSayValidator(TTSValidator):

    def __init__(self, tts):
        if False:
            return 10
        super(SpdSayValidator, self).__init__(tts)

    def validate_lang(self):
        if False:
            while True:
                i = 10
        pass

    def validate_connection(self):
        if False:
            i = 10
            return i + 15
        try:
            subprocess.call(['spd-say', '--version'])
        except Exception:
            raise Exception('SpdSay is not installed. Run: sudo apt-get install speech-dispatcher')

    def get_tts_class(self):
        if False:
            i = 10
            return i + 15
        return SpdSay