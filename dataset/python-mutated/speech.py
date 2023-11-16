import re
import time
from threading import Lock
from mycroft.configuration import Configuration
from mycroft.metrics import report_timing, Stopwatch
from mycroft.tts import TTSFactory
from mycroft.util import check_for_signal
from mycroft.util.log import LOG
from mycroft.messagebus.message import Message
from mycroft.tts.remote_tts import RemoteTTSException
from mycroft.tts.mimic_tts import Mimic
bus = None
config = None
tts = None
tts_hash = None
lock = Lock()
mimic_fallback_obj = None
_last_stop_signal = 0

def handle_speak(event):
    if False:
        while True:
            i = 10
    'Handle "speak" message\n\n    Parse sentences and invoke text to speech service.\n    '
    config = Configuration.get()
    Configuration.set_config_update_handlers(bus)
    global _last_stop_signal
    event.context = event.context or {}
    if event.context.get('destination') and (not ('debug_cli' in event.context['destination'] or 'audio' in event.context['destination'])):
        return
    if event.context and 'ident' in event.context:
        ident = event.context['ident']
    else:
        ident = 'unknown'
    start = time.time()
    with lock:
        stopwatch = Stopwatch()
        stopwatch.start()
        utterance = event.data['utterance']
        listen = event.data.get('expect_response', False)
        if config.get('enclosure', {}).get('platform') != 'picroft' and len(re.findall('<[^>]*>', utterance)) == 0:
            chunks = tts.preprocess_utterance(utterance)
            chunks = [(chunks[i], listen if i == len(chunks) - 1 else False) for i in range(len(chunks))]
            for (chunk, listen) in chunks:
                if _last_stop_signal > start or check_for_signal('buttonPress'):
                    tts.playback.clear()
                    break
                try:
                    mute_and_speak(chunk, ident, listen)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    LOG.error('Error in mute_and_speak', exc_info=True)
        else:
            mute_and_speak(utterance, ident, listen)
        stopwatch.stop()
    report_timing(ident, 'speech', stopwatch, {'utterance': utterance, 'tts': tts.__class__.__name__})

def mute_and_speak(utterance, ident, listen=False):
    if False:
        i = 10
        return i + 15
    'Mute mic and start speaking the utterance using selected tts backend.\n\n    Args:\n        utterance:  The sentence to be spoken\n        ident:      Ident tying the utterance to the source query\n    '
    global tts_hash
    if tts_hash != hash(str(config.get('tts', ''))):
        global tts
        if tts:
            tts.playback.detach_tts(tts)
        tts = TTSFactory.create()
        tts.init(bus)
        tts_hash = hash(str(config.get('tts', '')))
    LOG.info('Speak: ' + utterance)
    try:
        tts.execute(utterance, ident, listen)
    except RemoteTTSException as e:
        LOG.error(e)
        mimic_fallback_tts(utterance, ident, listen)
    except Exception:
        LOG.exception('TTS execution failed.')

def _get_mimic_fallback():
    if False:
        i = 10
        return i + 15
    'Lazily initializes the fallback TTS if needed.'
    global mimic_fallback_obj
    if not mimic_fallback_obj:
        config = Configuration.get()
        tts_config = config.get('tts', {}).get('mimic', {})
        lang = config.get('lang', 'en-us')
        tts = Mimic(lang, tts_config)
        tts.validator.validate()
        tts.init(bus)
        mimic_fallback_obj = tts
    return mimic_fallback_obj

def mimic_fallback_tts(utterance, ident, listen):
    if False:
        for i in range(10):
            print('nop')
    'Speak utterance using fallback TTS if connection is lost.\n\n    Args:\n        utterance (str): sentence to speak\n        ident (str): interaction id for metrics\n        listen (bool): True if interaction should end with mycroft listening\n    '
    tts = _get_mimic_fallback()
    LOG.debug('Mimic fallback, utterance : ' + str(utterance))
    tts.execute(utterance, ident, listen)

def handle_stop(event):
    if False:
        while True:
            i = 10
    'Handle stop message.\n\n    Shutdown any speech.\n    '
    global _last_stop_signal
    if check_for_signal('isSpeaking', -1):
        _last_stop_signal = time.time()
        tts.playback.clear()
        bus.emit(Message('mycroft.stop.handled', {'by': 'TTS'}))

def init(messagebus):
    if False:
        for i in range(10):
            print('nop')
    'Start speech related handlers.\n\n    Args:\n        messagebus: Connection to the Mycroft messagebus\n    '
    global bus
    global tts
    global tts_hash
    global config
    bus = messagebus
    Configuration.set_config_update_handlers(bus)
    config = Configuration.get()
    bus.on('mycroft.stop', handle_stop)
    bus.on('mycroft.audio.speech.stop', handle_stop)
    bus.on('speak', handle_speak)
    tts = TTSFactory.create()
    tts.init(bus)
    tts_hash = hash(str(config.get('tts', '')))

def shutdown():
    if False:
        for i in range(10):
            print('nop')
    'Shutdown the audio service cleanly.\n\n    Stop any playing audio and make sure threads are joined correctly.\n    '
    if tts:
        tts.playback.stop()
        tts.playback.join()
    if mimic_fallback_obj:
        mimic_fallback_obj.playback.stop()
        mimic_fallback_obj.playback.join()