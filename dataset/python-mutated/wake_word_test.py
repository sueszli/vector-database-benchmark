import time
import wave
from glob import glob
import os
import pyee
from os.path import dirname, join
from speech_recognition import AudioSource
from mycroft.client.speech.listener import RecognizerLoop
from mycroft.client.speech.mic import ResponsiveRecognizer
'Tests for determining the accuracy of the selected wake word engine.'

def to_percent(val):
    if False:
        print('Hello World!')
    return '{0:.2f}'.format(100.0 * val) + '%'

class FileStream:
    MIN_S_TO_DEBUG = 5.0
    UPDATE_INTERVAL_S = 1.0

    def __init__(self, file_name):
        if False:
            return 10
        self.file = wave.open(file_name, 'rb')
        self.size = self.file.getnframes()
        self.sample_rate = self.file.getframerate()
        self.sample_width = self.file.getsampwidth()
        self.last_update_time = 0.0
        self.total_s = self.size / self.sample_rate / self.sample_width
        if self.total_s > self.MIN_S_TO_DEBUG:
            self.debug = True
        else:
            self.debug = False

    def calc_progress(self):
        if False:
            i = 10
            return i + 15
        return float(self.file.tell()) / self.size

    def read(self, chunk_size, of_exc=False):
        if False:
            return 10
        progress = self.calc_progress()
        if progress == 1.0:
            raise EOFError
        if self.debug:
            cur_time = time.time()
            dt = cur_time - self.last_update_time
            if dt > self.UPDATE_INTERVAL_S:
                self.last_update_time = cur_time
                print(to_percent(progress))
        return self.file.readframes(chunk_size)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.file.close()

class FileMockMicrophone(AudioSource):

    def __init__(self, file_name):
        if False:
            i = 10
            return i + 15
        self.stream = FileStream(file_name)
        self.SAMPLE_RATE = self.stream.sample_rate
        self.SAMPLE_WIDTH = self.stream.sample_width
        self.CHUNK = 1024
        self.muted = False

    def mute(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def unmute(self):
        if False:
            while True:
                i = 10
        pass

    def close(self):
        if False:
            print('Hello World!')
        self.stream.close()

    def duration_to_bytes(self, ww_duration):
        if False:
            print('Hello World!')
        return int(ww_duration * self.SAMPLE_RATE * self.SAMPLE_WIDTH)

class AudioTester:

    def __init__(self, samp_rate):
        if False:
            i = 10
            return i + 15
        print()
        self.ww_recognizer = RecognizerLoop().create_wake_word_recognizer()
        self.listener = ResponsiveRecognizer(self.ww_recognizer)
        self.listener.config['confirm_listening'] = False
        print()

    def test_audio(self, file_name):
        if False:
            i = 10
            return i + 15
        source = FileMockMicrophone(file_name)
        ee = pyee.EventEmitter()

        class SharedData:
            times_found = 0
            found = False

        def on_found_wake_word():
            if False:
                while True:
                    i = 10
            SharedData.times_found += 1
        ee.on('recognizer_loop:record_begin', on_found_wake_word)
        try:
            while True:
                self.listener.listen(source, ee)
        except EOFError:
            cnt = 0
            while cnt < 2.0:
                if SharedData.times_found > 0:
                    break
                else:
                    time.sleep(0.1)
                    cnt += 0.1
        return SharedData.times_found

class Color:
    BOLD = '\x1b[1m'
    NORMAL = '\x1b[0m'
    GREEN = '\x1b[92m'
    RED = '\x1b[91m'

def bold_str(val):
    if False:
        i = 10
        return i + 15
    return Color.BOLD + str(val) + Color.NORMAL

def get_root_dir():
    if False:
        for i in range(10):
            print('nop')
    return dirname(dirname(__file__))

def get_file_names(folder):
    if False:
        while True:
            i = 10
    query = join(folder, '*.wav')
    root_dir = get_root_dir()
    full_path = join(root_dir, query)
    file_names = sorted(glob(full_path))
    if len(file_names) < 1:
        raise IOError
    return file_names

def test_audio_files(tester, file_names, on_file_finish):
    if False:
        return 10
    num_found = 0
    for file_name in file_names:
        short_name = os.path.basename(file_name)
        times_found = tester.test_audio(file_name)
        num_found += times_found
        on_file_finish(short_name, times_found)
    return num_found

def file_frame_rate(file_name):
    if False:
        for i in range(10):
            print('nop')
    wf = wave.open(file_name, 'rb')
    frame_rate = wf.getframerate()
    wf.close()
    return frame_rate

def print_ww_found_status(word, short_name):
    if False:
        i = 10
        return i + 15
    print('Wake word ' + bold_str(word) + ' - ' + short_name)

def test_false_negative(directory):
    if False:
        for i in range(10):
            print('nop')
    file_names = get_file_names(directory)
    tester = AudioTester(file_frame_rate(file_names[0]))

    def on_file_finish(short_name, times_found):
        if False:
            for i in range(10):
                print('nop')
        not_found_str = Color.RED + 'Not found'
        found_str = Color.GREEN + 'Detected '
        status_str = not_found_str if times_found == 0 else found_str
        print_ww_found_status(status_str, short_name)
    num_found = test_audio_files(tester, file_names, on_file_finish)
    total = len(file_names)
    print
    print('Found ' + bold_str(num_found) + ' out of ' + bold_str(total))
    print(bold_str(to_percent(float(num_found) / total)) + ' accuracy.')
    print

def test_false_positive(directory):
    if False:
        print('Hello World!')
    file_names = get_file_names(directory)
    tester = AudioTester(file_frame_rate(file_names[0]))

    def on_file_finish(short_name, times_found):
        if False:
            while True:
                i = 10
        not_found_str = Color.GREEN + 'Not found'
        found_str = Color.RED + 'Detected '
        status_str = not_found_str if times_found == 0 else found_str
        print_ww_found_status(status_str, short_name)
    num_found = test_audio_files(tester, file_names, on_file_finish)
    total = len(file_names)
    print
    print('Found ' + bold_str(num_found) + ' false positives')
    print('in ' + bold_str(str(total)) + ' files')
    print

def run_test():
    if False:
        for i in range(10):
            print('nop')
    directory = join(dirname(__file__), 'data')
    false_neg_dir = join(directory, 'with_wake_word')
    false_pos_dir = join(directory, 'without_wake_word')
    try:
        test_false_negative(false_neg_dir)
    except IOError:
        print(bold_str('Warning: No wav files found in ' + false_neg_dir))
    try:
        test_false_positive(false_pos_dir)
    except IOError:
        print(bold_str('Warning: No wav files found in ' + false_pos_dir))
    print('Complete!')