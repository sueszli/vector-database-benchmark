from mycroft.tts.tts import TTS, TTSValidator
from mycroft.configuration import Configuration

class PollyTTS(TTS):

    def __init__(self, lang='en-us', config=None):
        if False:
            while True:
                i = 10
        import boto3
        config = config or Configuration.get().get('tts', {}).get('polly', {})
        super(PollyTTS, self).__init__(lang, config, PollyTTSValidator(self), audio_ext='mp3', ssml_tags=['speak', 'say-as', 'voice', 'prosody', 'break', 'emphasis', 'sub', 'lang', 'phoneme', 'w', 'whisper', 'amazon:auto-breaths', 'p', 's', 'amazon:effect', 'mark'])
        self.voice = self.config.get('voice', 'Matthew')
        self.key_id = self.config.get('access_key_id', '')
        self.key = self.config.get('secret_access_key', '')
        self.region = self.config.get('region', 'us-east-1')
        self.engine = self.config.get('engine', 'standard')
        self.polly = boto3.Session(aws_access_key_id=self.key_id, aws_secret_access_key=self.key, region_name=self.region).client('polly')

    def get_tts(self, sentence, wav_file):
        if False:
            return 10
        text_type = 'text'
        if self.remove_ssml(sentence) != sentence:
            text_type = 'ssml'
            sentence = sentence.replace('\\whispered', '/amazon:effect').replace('whispered', 'amazon:effect name="whispered"')
        response = self.polly.synthesize_speech(OutputFormat=self.audio_ext, Text=sentence, TextType=text_type, VoiceId=self.voice, Engine=self.engine)
        with open(wav_file, 'wb') as f:
            f.write(response['AudioStream'].read())
        return (wav_file, None)

    def describe_voices(self, language_code='en-US'):
        if False:
            for i in range(10):
                print('nop')
        if language_code.islower():
            (a, b) = language_code.split('-')
            b = b.upper()
            language_code = '-'.join([a, b])
        voices = self.polly.describe_voices(LanguageCode=language_code)
        return voices

class PollyTTSValidator(TTSValidator):

    def __init__(self, tts):
        if False:
            while True:
                i = 10
        super(PollyTTSValidator, self).__init__(tts)

    def validate_lang(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def validate_dependencies(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            from boto3 import Session
        except ImportError:
            raise Exception('PollyTTS dependencies not installed, please run pip install boto3 ')

    def validate_connection(self):
        if False:
            i = 10
            return i + 15
        try:
            if not self.tts.voice:
                raise Exception('Polly TTS Voice not configured')
            output = self.tts.describe_voices()
        except TypeError:
            raise Exception('PollyTTS server could not be verified. Please check your internet connection and credentials.')

    def get_tts_class(self):
        if False:
            return 10
        return PollyTTS