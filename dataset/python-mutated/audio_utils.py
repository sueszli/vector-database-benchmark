"""Contains simple tools for performing audio related tasks such as playback
of audio, recording and listing devices.
"""
from copy import deepcopy
import os
import pyaudio
import re
import subprocess
import mycroft.configuration
from .log import LOG

def play_audio_file(uri: str, environment=None):
    if False:
        print('Hello World!')
    'Play an audio file.\n\n    This wraps the other play_* functions, choosing the correct one based on\n    the file extension. The function will return directly and play the file\n    in the background.\n\n    Args:\n        uri:    uri to play\n        environment (dict): optional environment for the subprocess call\n\n    Returns: subprocess.Popen object. None if the format is not supported or\n             an error occurs playing the file.\n    '
    extension_to_function = {'.wav': play_wav, '.mp3': play_mp3, '.ogg': play_ogg}
    (_, extension) = os.path.splitext(uri)
    play_function = extension_to_function.get(extension.lower())
    if play_function:
        return play_function(uri, environment)
    else:
        LOG.error('Could not find a function capable of playing {uri}. Supported formats are {keys}.'.format(uri=uri, keys=list(extension_to_function.keys())))
        return None
_ENVIRONMENT = deepcopy(os.environ)
_ENVIRONMENT['PULSE_PROP'] = 'media.role=music'

def _get_pulse_environment(config):
    if False:
        return 10
    'Return environment for pulse audio depeding on ducking config.'
    tts_config = config.get('tts', {})
    if tts_config and tts_config.get('pulse_duck'):
        return _ENVIRONMENT
    else:
        return os.environ

def _play_cmd(cmd, uri, config, environment):
    if False:
        for i in range(10):
            print('nop')
    'Generic function for starting playback from a commandline and uri.\n\n    Args:\n        cmd (str): commandline to execute %1 in the command line will be\n                   replaced with the provided uri.\n        uri (str): uri to play\n        config (dict): config to use\n        environment: environment to execute in, can be used to supply specific\n                     pulseaudio settings.\n    '
    environment = environment or _get_pulse_environment(config)
    cmd_elements = str(cmd).split(' ')
    cmdline = [e if e != '%1' else uri for e in cmd_elements]
    return subprocess.Popen(cmdline, env=environment)

def play_wav(uri, environment=None):
    if False:
        while True:
            i = 10
    'Play a wav-file.\n\n    This will use the application specified in the mycroft config\n    and play the uri passed as argument. The function will return directly\n    and play the file in the background.\n\n    Args:\n        uri:    uri to play\n        environment (dict): optional environment for the subprocess call\n\n    Returns: subprocess.Popen object or None if operation failed\n    '
    config = mycroft.configuration.Configuration.get()
    play_wav_cmd = config['play_wav_cmdline']
    try:
        return _play_cmd(play_wav_cmd, uri, config, environment)
    except FileNotFoundError as e:
        LOG.error('Failed to launch WAV: {} ({})'.format(play_wav_cmd, repr(e)))
    except Exception:
        LOG.exception('Failed to launch WAV: {}'.format(play_wav_cmd))
    return None

def play_mp3(uri, environment=None):
    if False:
        print('Hello World!')
    'Play a mp3-file.\n\n    This will use the application specified in the mycroft config\n    and play the uri passed as argument. The function will return directly\n    and play the file in the background.\n\n    Args:\n        uri:    uri to play\n        environment (dict): optional environment for the subprocess call\n\n    Returns: subprocess.Popen object or None if operation failed\n    '
    config = mycroft.configuration.Configuration.get()
    play_mp3_cmd = config.get('play_mp3_cmdline')
    try:
        return _play_cmd(play_mp3_cmd, uri, config, environment)
    except FileNotFoundError as e:
        LOG.error('Failed to launch MP3: {} ({})'.format(play_mp3_cmd, repr(e)))
    except Exception:
        LOG.exception('Failed to launch MP3: {}'.format(play_mp3_cmd))
    return None

def play_ogg(uri, environment=None):
    if False:
        i = 10
        return i + 15
    'Play an ogg-file.\n\n    This will use the application specified in the mycroft config\n    and play the uri passed as argument. The function will return directly\n    and play the file in the background.\n\n    Args:\n        uri:    uri to play\n        environment (dict): optional environment for the subprocess call\n\n    Returns: subprocess.Popen object, or None if operation failed\n    '
    config = mycroft.configuration.Configuration.get()
    play_ogg_cmd = config.get('play_ogg_cmdline')
    try:
        return _play_cmd(play_ogg_cmd, uri, config, environment)
    except FileNotFoundError as e:
        LOG.error('Failed to launch OGG: {} ({})'.format(play_ogg_cmd, repr(e)))
    except Exception:
        LOG.exception('Failed to launch OGG: {}'.format(play_ogg_cmd))
    return None

def record(file_path, duration, rate, channels):
    if False:
        for i in range(10):
            print('nop')
    'Simple function to record from the default mic.\n\n    The recording is done in the background by the arecord commandline\n    application.\n\n    Args:\n        file_path: where to store the recorded data\n        duration: how long to record\n        rate: sample rate\n        channels: number of channels\n\n    Returns:\n        process for performing the recording.\n    '
    command = ['arecord', '-r', str(rate), '-c', str(channels)]
    command += ['-d', str(duration)] if duration > 0 else []
    command += [file_path]
    return subprocess.Popen(command)

def find_input_device(device_name):
    if False:
        i = 10
        return i + 15
    "Find audio input device by name.\n\n    Args:\n        device_name: device name or regex pattern to match\n\n    Returns: device_index (int) or None if device wasn't found\n    "
    LOG.info('Searching for input device: {}'.format(device_name))
    LOG.debug('Devices: ')
    pa = pyaudio.PyAudio()
    pattern = re.compile(device_name)
    for device_index in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(device_index)
        LOG.debug('   {}'.format(dev['name']))
        if dev['maxInputChannels'] > 0 and pattern.match(dev['name']):
            LOG.debug('    ^-- matched')
            return device_index
    return None