"""
Cache handler - reads all the .dialog files (The default
mycroft responses) and does a tts inference.
It then saves the .wav files to mark1 device

* * * *   D E P R E C A T E D   * * * *
THIS MODULE IS DEPRECATED IN FAVOR OF tts/cache.py. IT WILL BE REMOVED
IN THE NEXT MAJOR RELEASE, 21.08

"""
import base64
import glob
import os
import re
import shutil
import hashlib
import json
import mycroft.util as util
from urllib import parse
from requests_futures.sessions import FuturesSession
from mycroft.util.log import LOG
REGEX_SPL_CHARS = re.compile('[@#$%^*()<>/\\|}{~:]')
MIMIC2_URL = 'https://mimic-api.mycroft.ai/synthesize?text='
TTS = 'Mimic2'
res_path = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', 'res', 'text', 'en-us'))
wifi_setup_path = '/usr/local/mycroft/mycroft-wifi-setup/dialog/en-us'
cache_dialog_path = [res_path, wifi_setup_path]

def generate_cache_text(cache_audio_dir, cache_text_file):
    if False:
        i = 10
        return i + 15
    '\n    This prepares a text file with all the sentences\n    from *.dialog files present in\n    mycroft/res/text/en-us and mycroft-wifi setup skill\n    Args:\n        cache_audio_dir (path): DEPRECATED path to store .wav files\n        cache_text_file (file): file containing the sentences\n    '
    if cache_audio_dir is not None:
        LOG.warning('the cache_audio_dir argument is deprecated. ensure the directory exists before executing this function. support for this argument will be removed in version 21.08')
        if not os.path.exists(cache_audio_dir):
            os.makedirs(cache_audio_dir)
    try:
        if not os.path.isfile(cache_text_file):
            text_file = open(cache_text_file, 'w')
            for each_path in cache_dialog_path:
                if os.path.exists(each_path):
                    write_cache_text(each_path, text_file)
            text_file.close()
            LOG.info('Completed generating cache')
        else:
            LOG.info("Cache file 'cache_text.txt' already exists")
    except Exception:
        LOG.exception('Could not open text file to write cache')

def write_cache_text(cache_path, f):
    if False:
        i = 10
        return i + 15
    for file in glob.glob(cache_path + '/*.dialog'):
        try:
            with open(file, 'r') as fp:
                all_dialogs = fp.readlines()
                for each_dialog in all_dialogs:
                    each_dialog = re.split('(?<!\\w\\.\\w.)(?<![A-Z][a-z]\\.)(?<=\\.|\\;|\\?)\\s', each_dialog.strip())
                    for each in each_dialog:
                        if REGEX_SPL_CHARS.search(each) is None:
                            f.write(each.strip() + '\n')
        except Exception:
            pass

def download_audio(cache_audio_dir, cache_text_file):
    if False:
        i = 10
        return i + 15
    "\n    This method takes the sentences from the text file generated\n    using generate_cache_text() and performs TTS inference on\n    mimic2-api. The wav files and phonemes are stored in\n    'cache_audio_dir'\n    Args:\n        cache_audio_dir (path): path to store .wav files\n        cache_text_file (file): file containing the sentences\n    "
    if os.path.isfile(cache_text_file) and os.path.exists(cache_audio_dir):
        if not os.listdir(cache_audio_dir):
            session = FuturesSession()
            with open(cache_text_file, 'r') as fp:
                all_dialogs = fp.readlines()
                for each_dialog in all_dialogs:
                    each_dialog = each_dialog.strip()
                    key = str(hashlib.md5(each_dialog.encode('utf-8', 'ignore')).hexdigest())
                    wav_file = os.path.join(cache_audio_dir, key + '.wav')
                    each_dialog = parse.quote(each_dialog)
                    mimic2_url = MIMIC2_URL + each_dialog + '&visimes=True'
                    try:
                        req = session.get(mimic2_url)
                        results = req.result().json()
                        audio = base64.b64decode(results['audio_base64'])
                        vis = results['visimes']
                        if audio:
                            with open(wav_file, 'wb') as audiofile:
                                audiofile.write(audio)
                        if vis:
                            pho_file = os.path.join(cache_audio_dir, key + '.pho')
                            with open(pho_file, 'w') as cachefile:
                                cachefile.write(json.dumps(vis))
                    except Exception:
                        LOG.exception('Unable to get pre-loaded cache')
            LOG.info('Completed getting cache for {}'.format(TTS))
        else:
            LOG.info('Pre-loaded cache for {} already exists'.format(TTS))
    else:
        missing_path = cache_text_file if not os.path.isfile(cache_text_file) else cache_audio_dir
        LOG.error('Path ({}) does not exist for getting the cache'.format(missing_path))

def copy_cache(cache_audio_dir):
    if False:
        for i in range(10):
            print('nop')
    "\n    This method copies the cache from 'cache_audio_dir'\n    to TTS specific cache directory given by\n    get_cache_directory()\n    Args:\n        cache_audio_dir (path): path containing .wav files\n    "
    if os.path.exists(cache_audio_dir):
        dest = util.get_cache_directory('tts/' + 'Mimic2')
        files = os.listdir(cache_audio_dir)
        for f in files:
            shutil.copy2(os.path.join(cache_audio_dir, f), dest)
        LOG.info('Copied all pre-loaded cache for {} to {}'.format(TTS, dest))
    else:
        LOG.info('No Source directory for {} pre-loaded cache'.format(TTS))

def main(cache_audio_dir):
    if False:
        for i in range(10):
            print('nop')
    if cache_audio_dir:
        if not os.path.exists(cache_audio_dir):
            os.makedirs(cache_audio_dir)
        cache_text_dir = os.path.dirname(cache_audio_dir)
        cache_text_path = os.path.join(cache_text_dir, 'cache_text.txt')
        generate_cache_text(None, cache_text_path)
        download_audio(cache_audio_dir, cache_text_path)
        copy_cache(cache_audio_dir)