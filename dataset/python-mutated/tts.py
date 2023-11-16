from copy import deepcopy
import os
import random
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
from threading import Thread
from time import time
from warnings import warn
import os.path
from os.path import dirname, exists, isdir, join
import mycroft.util
from mycroft.enclosure.api import EnclosureAPI
from mycroft.configuration import Configuration
from mycroft.messagebus.message import Message
from mycroft.metrics import report_timing, Stopwatch
from mycroft.util import play_wav, play_mp3, check_for_signal, create_signal, resolve_resource_file
from mycroft.util.file_utils import get_temp_path
from mycroft.util.log import LOG
from mycroft.util.plugins import load_plugin
from queue import Queue, Empty
from .cache import hash_sentence, TextToSpeechCache
_TTS_ENV = deepcopy(os.environ)
_TTS_ENV['PULSE_PROP'] = 'media.role=phone'
EMPTY_PLAYBACK_QUEUE_TUPLE = (None, None, None, None, None)
SSML_TAGS = re.compile('<[^>]*>')
WHITESPACE_AFTER_PERIOD = re.compile('\\b([A-za-z][\\.])(\\s+)')
SENTENCE_DELIMITERS = re.compile('(?<!\\w\\.\\w.)(?<![A-Z][a-z]\\.)(?<=\\.|\\;|\\?)\\s')

def default_preprocess_utterance(utterance):
    if False:
        while True:
            i = 10
    'Default method for preprocessing Mycroft utterances for TTS.\n\n    Args:\n        utteance (str): Input utterance\n\n    Returns:\n        [str]: list of preprocessed sentences\n    '
    utterance = WHITESPACE_AFTER_PERIOD.sub('\\g<1>', utterance)
    chunks = SENTENCE_DELIMITERS.split(utterance)
    return chunks

class PlaybackThread(Thread):
    """Thread class for playing back tts audio and sending
    viseme data to enclosure.
    """

    def __init__(self, queue):
        if False:
            i = 10
            return i + 15
        super(PlaybackThread, self).__init__()
        self.queue = queue
        self.tts = []
        self.bus = None
        self._terminated = False
        self._processing_queue = False
        self.enclosure = None
        self.p = None
        if Configuration.get().get('tts', {}).get('pulse_duck'):
            self.pulse_env = _TTS_ENV
        else:
            self.pulse_env = None

    def init(self, tts):
        if False:
            print('Hello World!')
        'DEPRECATED! Init the TTS Playback thread.\n\n        TODO: 22.02 Remove this\n        '
        self.attach_tts(tts)
        self.set_bus(tts.bus)

    def set_bus(self, bus):
        if False:
            return 10
        'Provide bus instance to the TTS Playback thread.\n\n        Args:\n            bus (MycroftBusClient): bus client\n        '
        self.bus = bus

    def attach_tts(self, tts):
        if False:
            i = 10
            return i + 15
        'Add TTS to be cache checked.'
        self.tts.append(tts)

    def detach_tts(self, tts):
        if False:
            i = 10
            return i + 15
        'Remove TTS from cache check.'
        self.tts.remove(tts)

    def clear_queue(self):
        if False:
            while True:
                i = 10
        'Remove all pending playbacks.'
        while not self.queue.empty():
            self.queue.get()
        try:
            self.p.terminate()
        except Exception:
            pass

    def run(self):
        if False:
            while True:
                i = 10
        "Thread main loop. Get audio and extra data from queue and play.\n\n        The queue messages is a tuple containing\n        snd_type: 'mp3' or 'wav' telling the loop what format the data is in\n        data: path to temporary audio data\n        videmes: list of visemes to display while playing\n        listen: if listening should be triggered at the end of the sentence.\n\n        Playback of audio is started and the visemes are sent over the bus\n        the loop then wait for the playback process to finish before starting\n        checking the next position in queue.\n\n        If the queue is empty the end_audio() is called possibly triggering\n        listening.\n        "
        while not self._terminated:
            try:
                (snd_type, data, visemes, ident, listen) = self.queue.get(timeout=2)
                self.blink(0.5)
                if not self._processing_queue:
                    self._processing_queue = True
                    self.begin_audio()
                stopwatch = Stopwatch()
                with stopwatch:
                    if snd_type == 'wav':
                        self.p = play_wav(data, environment=self.pulse_env)
                    elif snd_type == 'mp3':
                        self.p = play_mp3(data, environment=self.pulse_env)
                    if visemes:
                        self.show_visemes(visemes)
                    if self.p:
                        self.p.communicate()
                        self.p.wait()
                report_timing(ident, 'speech_playback', stopwatch)
                if self.queue.empty():
                    self.end_audio(listen)
                    self._processing_queue = False
                self.blink(0.2)
            except Empty:
                pass
            except Exception as e:
                LOG.exception(e)
                if self._processing_queue:
                    self.end_audio(listen)
                    self._processing_queue = False

    def begin_audio(self):
        if False:
            print('Hello World!')
        'Perform befining of speech actions.'
        if self.bus:
            self.bus.emit(Message('recognizer_loop:audio_output_start'))
        else:
            LOG.warning('Speech started before bus was attached.')

    def end_audio(self, listen):
        if False:
            for i in range(10):
                print('nop')
        "Perform end of speech output actions.\n\n        Will inform the system that speech has ended and trigger the TTS's\n        cache checks. Listening will be triggered if requested.\n\n        Args:\n            listen (bool): True if listening event should be emitted\n        "
        if self.bus:
            self.bus.emit(Message('recognizer_loop:audio_output_end'))
            if listen:
                self.bus.emit(Message('mycroft.mic.listen'))
            for tts in self.tts:
                tts.cache.curate()
            check_for_signal('isSpeaking')
        else:
            LOG.warning('Speech started before bus was attached.')

    def show_visemes(self, pairs):
        if False:
            return 10
        'Send viseme data to enclosure\n\n        Args:\n            pairs (list): Visime and timing pair\n\n        Returns:\n            bool: True if button has been pressed.\n        '
        if self.enclosure:
            self.enclosure.mouth_viseme(time(), pairs)

    def clear(self):
        if False:
            print('Hello World!')
        'Clear all pending actions for the TTS playback thread.'
        self.clear_queue()

    def blink(self, rate=1.0):
        if False:
            return 10
        "Blink mycroft's eyes"
        if self.enclosure and random.random() < rate:
            self.enclosure.eyes_blink('b')

    def stop(self):
        if False:
            while True:
                i = 10
        'Stop thread'
        self._terminated = True
        self.clear_queue()

class TTS(metaclass=ABCMeta):
    """TTS abstract class to be implemented by all TTS engines.

    It aggregates the minimum required parameters and exposes
    ``execute(sentence)`` and ``validate_ssml(sentence)`` functions.

    Args:
        lang (str):
        config (dict): Configuration for this specific tts engine
        validator (TTSValidator): Used to verify proper installation
        phonetic_spelling (bool): Whether to spell certain words phonetically
        ssml_tags (list): Supported ssml properties. Ex. ['speak', 'prosody']
    """
    queue = None
    playback = None

    def __init__(self, lang, config, validator, audio_ext='wav', phonetic_spelling=True, ssml_tags=None):
        if False:
            i = 10
            return i + 15
        super(TTS, self).__init__()
        self.bus = None
        self.lang = lang or 'en-us'
        self.config = config
        self.validator = validator
        self.phonetic_spelling = phonetic_spelling
        self.audio_ext = audio_ext
        self.ssml_tags = ssml_tags or []
        self.voice = config.get('voice')
        self.filename = get_temp_path('tts.wav')
        self.enclosure = None
        random.seed()
        if TTS.queue is None:
            TTS.queue = Queue()
            TTS.playback = PlaybackThread(TTS.queue)
            TTS.playback.start()
        self.spellings = self.load_spellings()
        self.tts_name = type(self).__name__
        self.cache = TextToSpeechCache(self.config, self.tts_name, self.audio_ext)
        self.cache.clear()

    @property
    def available_languages(self) -> set:
        if False:
            return 10
        'Return languages supported by this TTS implementation in this state\n\n        This property should be overridden by the derived class to advertise\n        what languages that engine supports.\n\n        Returns:\n            set: supported languages\n        '
        return set()

    def load_spellings(self):
        if False:
            while True:
                i = 10
        'Load phonetic spellings of words as dictionary.'
        path = join('text', self.lang.lower(), 'phonetic_spellings.txt')
        spellings_file = resolve_resource_file(path)
        if not spellings_file:
            return {}
        try:
            with open(spellings_file) as f:
                lines = filter(bool, f.read().split('\n'))
            lines = [i.split(':') for i in lines]
            return {key.strip(): value.strip() for (key, value) in lines}
        except ValueError:
            LOG.exception('Failed to load phonetic spellings.')
            return {}

    def begin_audio(self):
        if False:
            while True:
                i = 10
        'Helper function for child classes to call in execute().'
        self.bus.emit(Message('recognizer_loop:audio_output_start'))

    def end_audio(self, listen=False):
        if False:
            for i in range(10):
                print('nop')
        'Helper function for child classes to call in execute().\n\n        Sends the recognizer_loop:audio_output_end message (indicating\n        that speaking is done for the moment) as well as trigger listening\n        if it has been requested. It also checks if cache directory needs\n        cleaning to free up disk space.\n\n        Args:\n            listen (bool): indication if listening trigger should be sent.\n        '
        self.bus.emit(Message('recognizer_loop:audio_output_end'))
        if listen:
            self.bus.emit(Message('mycroft.mic.listen'))
        self.cache.curate()
        check_for_signal('isSpeaking')

    def init(self, bus):
        if False:
            i = 10
            return i + 15
        'Performs intial setup of TTS object.\n\n        Args:\n            bus:    Mycroft messagebus connection\n        '
        self.bus = bus
        TTS.playback.set_bus(bus)
        TTS.playback.attach_tts(self)
        self.enclosure = EnclosureAPI(self.bus)
        TTS.playback.enclosure = self.enclosure

    def get_tts(self, sentence, wav_file):
        if False:
            return 10
        'Abstract method that a tts implementation needs to implement.\n\n        Should get data from tts.\n\n        Args:\n            sentence(str): Sentence to synthesize\n            wav_file(str): output file\n\n        Returns:\n            tuple: (wav_file, phoneme)\n        '
        pass

    def modify_tag(self, tag):
        if False:
            return 10
        'Override to modify each supported ssml tag.\n\n        Args:\n            tag (str): SSML tag to check and possibly transform.\n        '
        return tag

    @staticmethod
    def remove_ssml(text):
        if False:
            print('Hello World!')
        'Removes SSML tags from a string.\n\n        Args:\n            text (str): input string\n\n        Returns:\n            str: input string stripped from tags.\n        '
        return re.sub('<[^>]*>', '', text).replace('  ', ' ')

    def validate_ssml(self, utterance):
        if False:
            while True:
                i = 10
        'Check if engine supports ssml, if not remove all tags.\n\n        Remove unsupported / invalid tags\n\n        Args:\n            utterance (str): Sentence to validate\n\n        Returns:\n            str: validated_sentence\n        '
        if not self.ssml_tags:
            return self.remove_ssml(utterance)
        tags = SSML_TAGS.findall(utterance)
        for tag in tags:
            if any((supported in tag for supported in self.ssml_tags)):
                utterance = utterance.replace(tag, self.modify_tag(tag))
            else:
                utterance = utterance.replace(tag, '')
        return utterance.replace('  ', ' ')

    def preprocess_utterance(self, utterance):
        if False:
            while True:
                i = 10
        'Preprocess utterance into list of chunks suitable for the TTS.\n\n        Perform general chunking and TTS specific chunking.\n        '
        chunks = default_preprocess_utterance(utterance)
        result = []
        for chunk in chunks:
            result += self._preprocess_sentence(chunk)
        return result

    def _preprocess_sentence(self, sentence):
        if False:
            print('Hello World!')
        'Default preprocessing is no preprocessing.\n\n        This method can be overridden to create chunks suitable to the\n        TTS engine in question.\n\n        Args:\n            sentence (str): sentence to preprocess\n\n        Returns:\n            list: list of sentence parts\n        '
        return [sentence]

    def execute(self, sentence, ident=None, listen=False):
        if False:
            for i in range(10):
                print('nop')
        'Convert sentence to speech, preprocessing out unsupported ssml\n\n        The method caches results if possible using the hash of the\n        sentence.\n\n        Args:\n            sentence: (str) Sentence to be spoken\n            ident: (str) Id reference to current interaction\n            listen: (bool) True if listen should be triggered at the end\n                    of the utterance.\n        '
        sentence = self.validate_ssml(sentence)
        create_signal('isSpeaking')
        self._execute(sentence, ident, listen)

    def _execute(self, sentence, ident, listen):
        if False:
            i = 10
            return i + 15
        if self.phonetic_spelling:
            for word in re.findall("[\\w']+", sentence):
                if word.lower() in self.spellings:
                    sentence = sentence.replace(word, self.spellings[word.lower()])
        chunks = self._preprocess_sentence(sentence)
        chunks = [(chunks[i], listen if i == len(chunks) - 1 else False) for i in range(len(chunks))]
        for (sentence, l) in chunks:
            sentence_hash = hash_sentence(sentence)
            if sentence_hash in self.cache:
                (audio_file, phoneme_file) = self._get_sentence_from_cache(sentence_hash)
                if phoneme_file is None:
                    phonemes = None
                else:
                    phonemes = phoneme_file.load()
            else:
                audio_file = self.cache.define_audio_file(sentence_hash)
                (returned_file, phonemes) = self.get_tts(sentence, str(audio_file.path))
                returned_file = Path(returned_file)
                if returned_file != audio_file.path:
                    warn(DeprecationWarning(f'{self.tts_name} is saving files to a different path than requested. If you are the maintainer of this plugin, please adhere to the file path argument provided. Modified paths will be ignored in a future release.'))
                    audio_file.path = returned_file
                if phonemes:
                    phoneme_file = self.cache.define_phoneme_file(sentence_hash)
                    phoneme_file.save(phonemes)
                else:
                    phoneme_file = None
                self.cache.cached_sentences[sentence_hash] = (audio_file, phoneme_file)
            viseme = self.viseme(phonemes) if phonemes else None
            TTS.queue.put((self.audio_ext, str(audio_file.path), viseme, ident, l))

    def _get_sentence_from_cache(self, sentence_hash):
        if False:
            while True:
                i = 10
        cached_sentence = self.cache.cached_sentences[sentence_hash]
        (audio_file, phoneme_file) = cached_sentence
        LOG.info('Found {} in TTS cache'.format(audio_file.name))
        return (audio_file, phoneme_file)

    def viseme(self, phonemes):
        if False:
            return 10
        'Create visemes from phonemes.\n\n        May be implemented to convert TTS phonemes into Mycroft mouth\n        visuals.\n\n        Args:\n            phonemes (str): String with phoneme data\n\n        Returns:\n            list: visemes\n        '
        return None

    def clear_cache(self):
        if False:
            return 10
        'Remove all cached files.'
        LOG.warning('This method is deprecated, use TextToSpeechCache.clear')
        if not os.path.exists(mycroft.util.get_cache_directory('tts')):
            return
        for d in os.listdir(mycroft.util.get_cache_directory('tts')):
            dir_path = os.path.join(mycroft.util.get_cache_directory('tts'), d)
            if os.path.isdir(dir_path):
                for f in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, f)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            elif os.path.isfile(dir_path):
                os.unlink(dir_path)

    def save_phonemes(self, key, phonemes):
        if False:
            print('Hello World!')
        'Cache phonemes\n\n        Args:\n            key (str):        Hash key for the sentence\n            phonemes (str):   phoneme string to save\n        '
        LOG.warning('This method is deprecated, use PhonemeFile.save')
        cache_dir = mycroft.util.get_cache_directory('tts/' + self.tts_name)
        pho_file = os.path.join(cache_dir, key + '.pho')
        try:
            with open(pho_file, 'w') as cachefile:
                cachefile.write(phonemes)
        except Exception:
            LOG.exception('Failed to write {} to cache'.format(pho_file))
            pass

    def load_phonemes(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Load phonemes from cache file.\n\n        Args:\n            key (str): Key identifying phoneme cache\n        '
        LOG.warning('This method is deprecated, use PhonemeFile.load')
        pho_file = os.path.join(mycroft.util.get_cache_directory('tts/' + self.tts_name), key + '.pho')
        if os.path.exists(pho_file):
            try:
                with open(pho_file, 'r') as cachefile:
                    phonemes = cachefile.read().strip()
                return phonemes
            except Exception:
                LOG.debug('Failed to read .PHO from cache')
        return None

class TTSValidator(metaclass=ABCMeta):
    """TTS Validator abstract class to be implemented by all TTS engines.

    It exposes and implements ``validate(tts)`` function as a template to
    validate the TTS engines.
    """

    def __init__(self, tts):
        if False:
            print('Hello World!')
        self.tts = tts

    def validate(self):
        if False:
            i = 10
            return i + 15
        self.validate_dependencies()
        self.validate_instance()
        self.validate_filename()
        self.validate_lang()
        self.validate_connection()

    def validate_dependencies(self):
        if False:
            print('Hello World!')
        "Determine if all the TTS's external dependencies are satisfied."
        pass

    def validate_instance(self):
        if False:
            return 10
        clazz = self.get_tts_class()
        if not isinstance(self.tts, clazz):
            raise AttributeError('tts must be instance of ' + clazz.__name__)

    def validate_filename(self):
        if False:
            for i in range(10):
                print('nop')
        filename = self.tts.filename
        if not (filename and filename.endswith('.wav')):
            raise AttributeError('file: %s must be in .wav format!' % filename)
        dir_path = dirname(filename)
        if not (exists(dir_path) and isdir(dir_path)):
            raise AttributeError('filename: %s is not valid!' % filename)

    @abstractmethod
    def validate_lang(self):
        if False:
            return 10
        'Ensure the TTS supports current language.'

    @abstractmethod
    def validate_connection(self):
        if False:
            return 10
        "Ensure the TTS can connect to it's backend.\n\n        This can mean for example being able to launch the correct executable\n        or contact a webserver.\n        "

    @abstractmethod
    def get_tts_class(self):
        if False:
            return 10
        'Return TTS class that this validator is for.'

def load_tts_plugin(module_name):
    if False:
        while True:
            i = 10
    'Wrapper function for loading tts plugin.\n\n    Args:\n        (str) Mycroft tts module name from config\n    Returns:\n        class: found tts plugin class\n    '
    return load_plugin('mycroft.plugin.tts', module_name)

class TTSFactory:
    """Factory class instantiating the configured TTS engine.

    The factory can select between a range of built-in TTS engines and also
    from TTS engine plugins.
    """
    from mycroft.tts.festival_tts import Festival
    from mycroft.tts.espeak_tts import ESpeak
    from mycroft.tts.fa_tts import FATTS
    from mycroft.tts.google_tts import GoogleTTS
    from mycroft.tts.mary_tts import MaryTTS
    from mycroft.tts.mimic_tts import Mimic
    from mycroft.tts.spdsay_tts import SpdSay
    from mycroft.tts.bing_tts import BingTTS
    from mycroft.tts.ibm_tts import WatsonTTS
    from mycroft.tts.mimic2_tts import Mimic2
    from mycroft.tts.yandex_tts import YandexTTS
    from mycroft.tts.dummy_tts import DummyTTS
    from mycroft.tts.polly_tts import PollyTTS
    from mycroft.tts.mozilla_tts import MozillaTTS
    CLASSES = {'mimic': Mimic, 'mimic2': Mimic2, 'google': GoogleTTS, 'marytts': MaryTTS, 'fatts': FATTS, 'festival': Festival, 'espeak': ESpeak, 'spdsay': SpdSay, 'watson': WatsonTTS, 'bing': BingTTS, 'yandex': YandexTTS, 'polly': PollyTTS, 'mozilla': MozillaTTS, 'dummy': DummyTTS}

    @staticmethod
    def create():
        if False:
            for i in range(10):
                print('nop')
        'Factory method to create a TTS engine based on configuration.\n\n        The configuration file ``mycroft.conf`` contains a ``tts`` section with\n        the name of a TTS module to be read by this method.\n\n        "tts": {\n            "module": <engine_name>\n        }\n        '
        config = Configuration.get()
        lang = config.get('lang', 'en-us')
        tts_module = config.get('tts', {}).get('module', 'mimic')
        tts_config = config.get('tts', {}).get(tts_module, {})
        tts_lang = tts_config.get('lang', lang)
        try:
            if tts_module in TTSFactory.CLASSES:
                clazz = TTSFactory.CLASSES[tts_module]
            else:
                clazz = load_tts_plugin(tts_module)
                LOG.info('Loaded plugin {}'.format(tts_module))
            if clazz is None:
                raise ValueError('TTS module not found')
            tts = clazz(tts_lang, tts_config)
            tts.validator.validate()
        except Exception:
            if tts_module != 'mimic':
                LOG.exception("The selected TTS backend couldn't be loaded. Falling back to Mimic")
                clazz = TTSFactory.CLASSES.get('mimic')
                tts_config = config.get('tts', {}).get('mimic', {})
                tts = clazz(tts_lang, tts_config)
                tts.validator.validate()
            else:
                LOG.exception('The TTS could not be loaded.')
                raise
        return tts