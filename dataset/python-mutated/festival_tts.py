import subprocess
from .tts import TTS, TTSValidator

class Festival(TTS):

    def __init__(self, lang, config):
        if False:
            i = 10
            return i + 15
        super(Festival, self).__init__(lang, config, FestivalValidator(self))

    def execute(self, sentence, ident=None, listen=False):
        if False:
            i = 10
            return i + 15
        encoding = self.config.get('encoding', 'utf8')
        lang = self.config.get('lang', self.lang)
        text = subprocess.Popen(('echo', sentence), stdout=subprocess.PIPE)
        if encoding != 'utf8':
            convert_cmd = ('iconv', '-f', 'utf8', '-t', encoding)
            converted_text = subprocess.Popen(convert_cmd, stdin=text.stdout, stdout=subprocess.PIPE)
            text.wait()
            text = converted_text
        tts_cmd = ('festival', '--tts', '--language', lang)
        self.begin_audio()
        subprocess.call(tts_cmd, stdin=text.stdout)
        self.end_audio(listen)

class FestivalValidator(TTSValidator):

    def __init__(self, tts):
        if False:
            while True:
                i = 10
        super(FestivalValidator, self).__init__(tts)

    def validate_lang(self):
        if False:
            while True:
                i = 10
        pass

    def validate_connection(self):
        if False:
            return 10
        try:
            subprocess.call(['festival', '--version'])
        except Exception:
            raise Exception('Festival is missing. Run: sudo apt-get install festival')

    def get_tts_class(self):
        if False:
            return 10
        return Festival