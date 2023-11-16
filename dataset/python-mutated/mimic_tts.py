"""Mimic TTS, a local TTS backend.

This Backend uses the mimic executable to render text into speech.
"""
import os
import os.path
from os.path import exists, join, expanduser
import stat
import subprocess
from threading import Thread
from time import sleep
from mycroft import MYCROFT_ROOT_PATH
from mycroft.api import DeviceApi
from mycroft.configuration import Configuration
from mycroft.util.download import download
from mycroft.util.log import LOG
from .tts import TTS, TTSValidator

def get_mimic_binary():
    if False:
        while True:
            i = 10
    'Find the mimic binary, either from config or from PATH.\n\n    Returns:\n        (str) path of mimic executable\n    '
    config = Configuration.get().get('tts', {}).get('mimic')
    bin_ = config.get('path', os.path.join(MYCROFT_ROOT_PATH, 'mimic', 'bin', 'mimic'))
    if not os.path.isfile(bin_):
        import distutils.spawn
        bin_ = distutils.spawn.find_executable('mimic')
    return bin_

def get_subscriber_voices():
    if False:
        while True:
            i = 10
    'Get dict of mimic voices exclusive to subscribers.\n\n    Returns:\n        (dict) map of voices to custom Mimic executables.\n    '
    data_dir = expanduser(Configuration.get()['data_dir'])
    return {'trinity': join(data_dir, 'voices/mimic_tn')}

def download_subscriber_voices(selected_voice):
    if False:
        for i in range(10):
            print('nop')
    'Function to download all premium voices.\n\n    The function starts with the currently selected if applicable\n    '
    subscriber_voices = get_subscriber_voices()

    def make_executable(dest):
        if False:
            print('Hello World!')
        'Call back function to make the downloaded file executable.'
        LOG.info('Make executable new voice binary executable')
        file_stat = os.stat(dest)
        os.chmod(dest, file_stat.st_mode | stat.S_IEXEC)
    voice_file = subscriber_voices.get(selected_voice)
    if voice_file is not None and (not exists(voice_file)):
        LOG.info("Voice doesn't exist, downloading")
        url = DeviceApi().get_subscriber_voice_url(selected_voice)
        if url:
            dl_status = download(url, voice_file, make_executable)
            while not dl_status.done:
                sleep(1)
        else:
            LOG.debug('{} is not available for this architecture'.format(selected_voice))
    for voice in subscriber_voices:
        voice_file = subscriber_voices[voice]
        if not exists(voice_file):
            url = DeviceApi().get_subscriber_voice_url(voice)
            if url:
                dl_status = download(url, voice_file, make_executable)
                while not dl_status.done:
                    sleep(1)
            else:
                LOG.debug('{} is not available for this architecture'.format(voice))

def parse_phonemes(phonemes):
    if False:
        while True:
            i = 10
    'Parse mimic phoneme string into a list of phone, duration pairs.\n\n    Arguments\n        phonemes (bytes): phoneme output from mimic\n    Returns:\n        (list) list of phoneme duration pairs\n    '
    phon_str = phonemes.decode()
    pairs = phon_str.split(' ')
    return [pair.split(':') for pair in pairs if ':' in pair]

class Mimic(TTS):
    """TTS interface for local mimic v1."""

    def __init__(self, lang, config):
        if False:
            i = 10
            return i + 15
        super(Mimic, self).__init__(lang, config, MimicValidator(self), 'wav', ssml_tags=['speak', 'ssml', 'phoneme', 'voice', 'audio', 'prosody'])
        self.default_binary = get_mimic_binary()
        self.subscriber_voices = get_subscriber_voices()
        self.is_subscriber = DeviceApi().is_subscriber
        if self.is_subscriber:
            trd = Thread(target=download_subscriber_voices, args=[self.voice])
            trd.daemon = True
            trd.start()

    def modify_tag(self, tag):
        if False:
            return 10
        'Modify the SSML to suite Mimic.'
        ssml_conversions = {'x-slow': '0.4', 'slow': '0.7', 'medium': '1.0', 'high': '1.3', 'x-high': '1.6', 'speed': 'rate'}
        for (key, value) in ssml_conversions.items():
            tag = tag.replace(key, value)
        return tag

    @property
    def args(self):
        if False:
            i = 10
            return i + 15
        'Build mimic arguments.'
        subscriber_voices = self.subscriber_voices
        if self.voice in subscriber_voices and exists(subscriber_voices[self.voice]) and self.is_subscriber:
            mimic_bin = subscriber_voices[self.voice]
            voice = self.voice
        elif self.voice in subscriber_voices:
            mimic_bin = self.default_binary
            voice = 'ap'
        else:
            mimic_bin = self.default_binary
            voice = self.voice
        args = [mimic_bin, '-voice', voice, '-psdur', '-ssml']
        stretch = self.config.get('duration_stretch', None)
        if stretch:
            args += ['--setf', 'duration_stretch={}'.format(stretch)]
        return args

    def get_tts(self, sentence, wav_file):
        if False:
            print('Hello World!')
        'Generate WAV and phonemes.\n\n        Args:\n            sentence (str): sentence to generate audio for\n            wav_file (str): output file\n\n        Returns:\n            tuple ((str) file location, (str) generated phonemes)\n        '
        phonemes = subprocess.check_output(self.args + ['-o', wav_file, '-t', sentence])
        return (wav_file, parse_phonemes(phonemes))

    def viseme(self, phoneme_pairs):
        if False:
            for i in range(10):
                print('nop')
        'Convert phoneme string to visemes.\n\n        Args:\n            phoneme_pairs (list): Phoneme output from mimic\n\n        Returns:\n            (list) list of tuples of viseme and duration\n        '
        visemes = []
        for (phon, dur) in phoneme_pairs:
            visemes.append((VISIMES.get(phon, '4'), float(dur)))
        return visemes

class MimicValidator(TTSValidator):
    """Validator class checking that Mimic can be used."""

    def validate_lang(self):
        if False:
            return 10
        'Verify that the language is supported.'

    def validate_connection(self):
        if False:
            return 10
        'Check that Mimic executable is found and works.'
        mimic_bin = get_mimic_binary()
        try:
            subprocess.call([mimic_bin, '--version'])
        except Exception as err:
            if mimic_bin:
                LOG.error('Failed to find mimic at: {}'.format(mimic_bin))
            else:
                LOG.error('Mimic executable not found')
            raise Exception('Mimic was not found. Run install-mimic.sh to install it.') from err

    def get_tts_class(self):
        if False:
            while True:
                i = 10
        'Return the TTS class associated with the validator.'
        return Mimic
VISIMES = {'v': '5', 'f': '5', 'uh': '2', 'w': '2', 'uw': '2', 'er': '2', 'r': '2', 'ow': '2', 'b': '4', 'p': '4', 'm': '4', 'aw': '1', 'th': '3', 'dh': '3', 'zh': '3', 'ch': '3', 'sh': '3', 'jh': '3', 'oy': '6', 'ao': '6', 'z': '3', 's': '3', 'ae': '0', 'eh': '0', 'ey': '0', 'ah': '0', 'ih': '0', 'y': '0', 'iy': '0', 'aa': '0', 'ay': '0', 'ax': '0', 'hh': '0', 'n': '3', 't': '3', 'd': '3', 'l': '3', 'g': '3', 'ng': '3', 'k': '3', 'pau': '4'}