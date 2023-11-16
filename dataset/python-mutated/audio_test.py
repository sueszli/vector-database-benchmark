import argparse
import os
import pyaudio
from contextlib import contextmanager
from speech_recognition import Recognizer
from mycroft.client.speech.mic import MutableMicrophone
from mycroft.configuration import Configuration
from mycroft.util.audio_utils import play_wav
from mycroft.util.log import LOG
import logging
from mycroft.util.file_utils import get_temp_path
'\nAudio Test\nA tool for recording X seconds of audio, and then playing them back. Useful\nfor testing hardware, and ensures\ncompatibility with mycroft recognizer loop code.\n'
LOG.level = 'ERROR'
logging.getLogger('urllib3').setLevel(logging.WARNING)

@contextmanager
def mute_output():
    if False:
        i = 10
        return i + 15
    ' Context manager blocking stdout and stderr completely.\n\n    Redirects stdout and stderr to dev-null and restores them on exit.\n    '
    null_fds = [os.open(os.devnull, os.O_RDWR) for i in range(2)]
    orig_fds = [os.dup(1), os.dup(2)]
    os.dup2(null_fds[0], 1)
    os.dup2(null_fds[1], 2)
    try:
        yield
    finally:
        os.dup2(orig_fds[0], 1)
        os.dup2(orig_fds[1], 2)
        for fd in null_fds + orig_fds:
            os.close(fd)

def record(filename, duration):
    if False:
        i = 10
        return i + 15
    mic = MutableMicrophone()
    recognizer = Recognizer()
    with mic as source:
        audio = recognizer.record(source, duration=duration)
        with open(filename, 'wb') as f:
            f.write(audio.get_wav_data())

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', dest='filename', default=get_temp_path('test.wav'), help='Filename for saved audio (Default:{}'.format(get_temp_path('test.wav')))
    parser.add_argument('-d', '--duration', dest='duration', type=int, default=10, help='Duration of recording in seconds (Default: 10)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Add extra output regarding the recording')
    parser.add_argument('-l', '--list', dest='show_devices', action='store_true', default=False, help='List all availabile input devices')
    args = parser.parse_args()
    if args.show_devices:
        print(' Initializing... ')
        pa = pyaudio.PyAudio()
        print(' ====================== Audio Devices ======================')
        print('  Index    Device Name')
        for device_index in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(device_index)
            if dev['maxInputChannels'] > 0:
                print('   {}:       {}'.format(device_index, dev['name']))
        print()
    config = Configuration.get()
    if 'device_name' in config['listener']:
        dev = config['listener']['device_name']
    elif 'device_index' in config['listener']:
        dev = 'Device at index {}'.format(config['listener']['device_index'])
    else:
        dev = 'Default device'
    samplerate = config['listener']['sample_rate']
    play_cmd = config['play_wav_cmdline'].replace('%1', 'WAV_FILE')
    print(' ========================== Info ===========================')
    print(' Input device: {} @ Sample rate: {} Hz'.format(dev, samplerate))
    print(' Playback commandline: {}'.format(play_cmd))
    print()
    print(' ===========================================================')
    print(' ==         STARTING TO RECORD, MAKE SOME NOISE!          ==')
    print(' ===========================================================')
    if not args.verbose:
        with mute_output():
            record(args.filename, args.duration)
    else:
        record(args.filename, args.duration)
    print(' ===========================================================')
    print(' ==           DONE RECORDING, PLAYING BACK...             ==')
    print(' ===========================================================')
    status = play_wav(args.filename).wait()
    if status:
        print('An error occured while playing back audio ({})'.format(status))
if __name__ == '__main__':
    main()