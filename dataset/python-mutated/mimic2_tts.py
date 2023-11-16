import base64
import json
import math
import os
import re
from urllib import parse
from requests_futures.sessions import FuturesSession
from requests.exceptions import ReadTimeout, ConnectionError, ConnectTimeout, HTTPError
from mycroft.util.file_utils import get_cache_directory
from mycroft.util.log import LOG
from .mimic_tts import VISIMES
from .tts import TTS, TTSValidator
from .remote_tts import RemoteTTSException, RemoteTTSTimeoutException
_max_sentence_size = 170

def _break_chunks(l, n):
    if False:
        i = 10
        return i + 15
    'Yield successive n-sized chunks\n\n    Args:\n        l (list): text (str) to split\n        chunk_size (int): chunk size\n    '
    for i in range(0, len(l), n):
        yield ' '.join(l[i:i + n])

def _split_by_chunk_size(text, chunk_size):
    if False:
        for i in range(10):
            print('nop')
    'Split text into word chunks by chunk_size size\n\n    Args:\n        text (str): text to split\n        chunk_size (int): chunk size\n\n    Returns:\n        list: list of text chunks\n    '
    text_list = text.split()
    if len(text_list) <= chunk_size:
        return [text]
    if chunk_size < len(text_list) < chunk_size * 2:
        return list(_break_chunks(text_list, int(math.ceil(len(text_list) / 2))))
    elif chunk_size * 2 < len(text_list) < chunk_size * 3:
        return list(_break_chunks(text_list, int(math.ceil(len(text_list) / 3))))
    elif chunk_size * 3 < len(text_list) < chunk_size * 4:
        return list(_break_chunks(text_list, int(math.ceil(len(text_list) / 4))))
    else:
        return list(_break_chunks(text_list, int(math.ceil(len(text_list) / 5))))

def _split_by_punctuation(chunks, puncs):
    if False:
        return 10
    'Splits text by various punctionations\n    e.g. hello, world => [hello, world]\n\n    Args:\n        chunks (list or str): text (str) to split\n        puncs (list): list of punctuations used to split text\n\n    Returns:\n        list: list with split text\n    '
    if isinstance(chunks, str):
        out = [chunks]
    else:
        out = chunks
    for punc in puncs:
        splits = []
        for t in out:
            splits += re.split('(?<!\\.\\S)' + punc + '\\s', t)
        out = splits
    return [t.strip() for t in out]

def _add_punctuation(text):
    if False:
        print('Hello World!')
    'Add punctuation at the end of each chunk.\n\n    Mimic2 expects some form of punctuation at the end of a sentence.\n    '
    punctuation = ['.', '?', '!', ';']
    if len(text) >= 1 and text[-1] not in punctuation:
        return text + '.'
    else:
        return text

def _sentence_chunker(text):
    if False:
        while True:
            i = 10
    'Split text into smaller chunks for TTS generation.\n\n    NOTE: The smaller chunks are needed due to current Mimic2 TTS limitations.\n    This stage can be removed once Mimic2 can generate longer sentences.\n\n    Args:\n        text (str): text to split\n        chunk_size (int): size of each chunk\n        split_by_punc (bool, optional): Defaults to True.\n\n    Returns:\n        list: list of text chunks\n    '
    if len(text) <= _max_sentence_size:
        return [_add_punctuation(text)]
    first_splits = _split_by_punctuation(text, puncs=['\\.', '\\!', '\\?', '\\:', '\\;'])
    second_splits = []
    for chunk in first_splits:
        if len(chunk) > _max_sentence_size:
            second_splits += _split_by_punctuation(chunk, puncs=['\\,', '--', '-'])
        else:
            second_splits.append(chunk)
    third_splits = []
    for chunk in second_splits:
        if len(chunk) > _max_sentence_size:
            third_splits += _split_by_chunk_size(chunk, 20)
        else:
            third_splits.append(chunk)
    return [_add_punctuation(chunk) for chunk in third_splits]

class Mimic2(TTS):
    """Interface to the Mimic2 TTS."""

    def __init__(self, lang, config):
        if False:
            print('Hello World!')
        super().__init__(lang, config, Mimic2Validator(self))
        self.cache.load_persistent_cache()
        self.url = config['url']
        self.session = FuturesSession()

    def _requests(self, sentence):
        if False:
            while True:
                i = 10
        'Create asynchronous request list\n\n        Args:\n            chunks (list): list of text to synthesize\n\n        Returns:\n            list: list of FutureSession objects\n        '
        url = self.url + parse.quote(sentence)
        req_route = url + '&visimes=True'
        return self.session.get(req_route, timeout=5)

    def viseme(self, phonemes):
        if False:
            while True:
                i = 10
        'Maps phonemes to appropriate viseme encoding\n\n        Args:\n            phonemes (list): list of tuples (phoneme, time_start)\n\n        Returns:\n            list: list of tuples (viseme_encoding, time_start)\n        '
        visemes = []
        for pair in phonemes:
            if pair[0]:
                phone = pair[0].lower()
            else:
                phone = 'z'
            vis = VISIMES.get(phone)
            vis_dur = float(pair[1])
            visemes.append((vis, vis_dur))
        return visemes

    def _preprocess_sentence(self, sentence):
        if False:
            for i in range(10):
                print('nop')
        'Split sentence in chunks better suited for mimic2. '
        return _sentence_chunker(sentence)

    def get_tts(self, sentence, wav_file):
        if False:
            print('Hello World!')
        'Generate (remotely) and play mimic2 WAV audio\n\n        Args:\n            sentence (str): Phrase to synthesize to audio with mimic2\n            wav_file (str): Location to write audio output\n        '
        LOG.debug('Generating Mimic2 TSS for: ' + str(sentence))
        try:
            res = self._requests(sentence).result()
            if 200 <= res.status_code < 300:
                results = res.json()
                audio = base64.b64decode(results['audio_base64'])
                vis = results['visimes']
                with open(wav_file, 'wb') as f:
                    f.write(audio)
            else:
                raise RemoteTTSException('Backend returned HTTP status {}'.format(res.status_code))
        except (ReadTimeout, ConnectionError, ConnectTimeout, HTTPError):
            raise RemoteTTSTimeoutException('Mimic 2 server request timed out. Falling back to mimic')
        return (wav_file, vis)

    def save_phonemes(self, key, phonemes):
        if False:
            while True:
                i = 10
        'Cache phonemes\n\n        Args:\n            key:        Hash key for the sentence\n            phonemes:   phoneme string to save\n        '
        cache_dir = get_cache_directory('tts/' + self.tts_name)
        pho_file = os.path.join(cache_dir, key + '.pho')
        try:
            with open(pho_file, 'w') as cachefile:
                cachefile.write(json.dumps(phonemes))
        except Exception:
            LOG.exception('Failed to write {} to cache'.format(pho_file))

    def load_phonemes(self, key):
        if False:
            print('Hello World!')
        'Load phonemes from cache file.\n\n        Args:\n            Key:    Key identifying phoneme cache\n        '
        pho_file = os.path.join(get_cache_directory('tts/' + self.tts_name), key + '.pho')
        if os.path.exists(pho_file):
            try:
                with open(pho_file, 'r') as cachefile:
                    phonemes = json.load(cachefile)
                return phonemes
            except Exception as e:
                LOG.error('Failed to read .PHO from cache ({})'.format(e))
        return None

class Mimic2Validator(TTSValidator):

    def __init__(self, tts):
        if False:
            while True:
                i = 10
        super(Mimic2Validator, self).__init__(tts)

    def validate_lang(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def validate_connection(self):
        if False:
            while True:
                i = 10
        pass

    def get_tts_class(self):
        if False:
            print('Hello World!')
        return Mimic2