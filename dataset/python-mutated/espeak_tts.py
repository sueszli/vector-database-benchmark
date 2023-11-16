import subprocess
from .tts import TTS, TTSValidator

class ESpeak(TTS):
    """TTS module for generating speech using ESpeak."""

    def __init__(self, lang, config):
        if False:
            for i in range(10):
                print('nop')
        super(ESpeak, self).__init__(lang, config, ESpeakValidator(self))

    def get_tts(self, sentence, wav_file):
        if False:
            i = 10
            return i + 15
        "Generate WAV from sentence, phonemes aren't supported.\n\n        Args:\n            sentence (str): sentence to generate audio for\n            wav_file (str): output file\n\n        Returns:\n            tuple ((str) file location, None)\n        "
        arguments = ['espeak', '-v', self.lang + '+' + self.voice]
        amplitude = self.config.get('amplitude')
        if amplitude:
            arguments.append('-a ' + amplitude)
        gap = self.config.get('gap')
        if gap:
            arguments.append('-g ' + gap)
        capital = self.config.get('capital')
        if capital:
            arguments.append('-k ' + capital)
        pitch = self.config.get('pitch')
        if pitch:
            arguments.append('-p ' + pitch)
        speed = self.config.get('speed')
        if speed:
            arguments.append('-s ' + speed)
        arguments.extend(['-w', wav_file, sentence])
        subprocess.call(arguments)
        return (wav_file, None)

class ESpeakValidator(TTSValidator):

    def __init__(self, tts):
        if False:
            i = 10
            return i + 15
        super(ESpeakValidator, self).__init__(tts)

    def validate_lang(self):
        if False:
            while True:
                i = 10
        pass

    def validate_connection(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            subprocess.call(['espeak', '--version'])
        except Exception:
            raise Exception('ESpeak is not installed. Please install it on your system and restart Mycroft.')

    def get_tts_class(self):
        if False:
            while True:
                i = 10
        return ESpeak