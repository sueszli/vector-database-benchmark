import audioop
from time import sleep, time as get_time
from collections import deque, namedtuple
import datetime
import json
import os
from os.path import isdir, join
import pyaudio
import requests
import speech_recognition
from hashlib import md5
from io import BytesIO, StringIO
from speech_recognition import Microphone, AudioSource, AudioData
from tempfile import gettempdir
from threading import Thread, Lock
from mycroft.api import DeviceApi
from mycroft.configuration import Configuration
from mycroft.session import SessionManager
from mycroft.util import check_for_signal, get_ipc_directory, resolve_resource_file, play_wav
from mycroft.util.log import LOG
from .data_structures import RollingMean, CyclicAudioBuffer
WakeWordData = namedtuple('WakeWordData', ['audio', 'found', 'stopped', 'end_audio'])

class MutableStream:

    def __init__(self, wrapped_stream, format, muted=False):
        if False:
            while True:
                i = 10
        assert wrapped_stream is not None
        self.wrapped_stream = wrapped_stream
        self.SAMPLE_WIDTH = pyaudio.get_sample_size(format)
        self.muted_buffer = b''.join([b'\x00' * self.SAMPLE_WIDTH])
        self.read_lock = Lock()
        self.muted = muted
        if muted:
            self.mute()

    def mute(self):
        if False:
            i = 10
            return i + 15
        'Stop the stream and set the muted flag.'
        with self.read_lock:
            self.muted = True
            self.wrapped_stream.stop_stream()

    def unmute(self):
        if False:
            for i in range(10):
                print('nop')
        'Start the stream and clear the muted flag.'
        with self.read_lock:
            self.muted = False
            self.wrapped_stream.start_stream()

    def read(self, size, of_exc=False):
        if False:
            while True:
                i = 10
        'Read data from stream.\n\n        Args:\n            size (int): Number of bytes to read\n            of_exc (bool): flag determining if the audio producer thread\n                           should throw IOError at overflows.\n\n        Returns:\n            (bytes) Data read from device\n        '
        frames = deque()
        remaining = size
        with self.read_lock:
            while remaining > 0:
                if self.muted:
                    return self.muted_buffer
                to_read = min(self.wrapped_stream.get_read_available(), remaining)
                if to_read <= 0:
                    sleep(0.01)
                    continue
                result = self.wrapped_stream.read(to_read, exception_on_overflow=of_exc)
                frames.append(result)
                remaining -= to_read
        input_latency = self.wrapped_stream.get_input_latency()
        if input_latency > 0.2:
            LOG.warning('High input latency: %f' % input_latency)
        audio = b''.join(list(frames))
        return audio

    def close(self):
        if False:
            return 10
        self.wrapped_stream.close()
        self.wrapped_stream = None

    def is_stopped(self):
        if False:
            print('Hello World!')
        try:
            return self.wrapped_stream.is_stopped()
        except Exception as e:
            LOG.error(repr(e))
            return True

    def stop_stream(self):
        if False:
            i = 10
            return i + 15
        return self.wrapped_stream.stop_stream()

class MutableMicrophone(Microphone):

    def __init__(self, device_index=None, sample_rate=16000, chunk_size=1024, mute=False):
        if False:
            while True:
                i = 10
        Microphone.__init__(self, device_index=device_index, sample_rate=sample_rate, chunk_size=chunk_size)
        self.muted = False
        if mute:
            self.mute()

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self._start()

    def _start(self):
        if False:
            while True:
                i = 10
        'Open the selected device and setup the stream.'
        assert self.stream is None, 'This audio source is already inside a context manager'
        self.audio = pyaudio.PyAudio()
        self.stream = MutableStream(self.audio.open(input_device_index=self.device_index, channels=1, format=self.format, rate=self.SAMPLE_RATE, frames_per_buffer=self.CHUNK, input=True), self.format, self.muted)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            while True:
                i = 10
        return self._stop()

    def _stop(self):
        if False:
            while True:
                i = 10
        'Stop and close an open stream.'
        try:
            if not self.stream.is_stopped():
                self.stream.stop_stream()
            self.stream.close()
        except Exception:
            LOG.exception('Failed to stop mic input stream')
        self.stream = None
        self.audio.terminate()

    def restart(self):
        if False:
            return 10
        'Shutdown input device and restart.'
        self._stop()
        self._start()

    def mute(self):
        if False:
            while True:
                i = 10
        self.muted = True
        if self.stream:
            self.stream.mute()

    def unmute(self):
        if False:
            return 10
        self.muted = False
        if self.stream:
            self.stream.unmute()

    def is_muted(self):
        if False:
            i = 10
            return i + 15
        return self.muted

    def duration_to_bytes(self, sec):
        if False:
            for i in range(10):
                print('nop')
        'Converts a duration in seconds to number of recorded bytes.\n\n        Args:\n            sec: number of seconds\n\n        Returns:\n            (int) equivalent number of bytes recorded by this Mic\n        '
        return int(sec * self.SAMPLE_RATE) * self.SAMPLE_WIDTH

def get_silence(num_bytes):
    if False:
        while True:
            i = 10
    return b'\x00' * num_bytes

class NoiseTracker:
    """Noise tracker, used to deterimine if an audio utterance is complete.

    The current implementation expects a number of loud chunks (not necessary
    in one continous sequence) followed by a short period of continous quiet
    audio data to be considered complete.

    Args:
        minimum (int): lower noise level will be threshold for "quiet" level
        maximum (int): ceiling of noise level
        sec_per_buffer (float): the length of each buffer used when updating
                                the tracker
        loud_time_limit (float): time in seconds of low noise to be considered
                                 a complete sentence
        silence_time_limit (float): time limit for silence to abort sentence
        silence_after_loud (float): time of silence to finalize the sentence.
                                    default 0.25 seconds.
    """

    def __init__(self, minimum, maximum, sec_per_buffer, loud_time_limit, silence_time_limit, silence_after_loud_time=0.25):
        if False:
            print('Hello World!')
        self.min_level = minimum
        self.max_level = maximum
        self.sec_per_buffer = sec_per_buffer
        self.num_loud_chunks = 0
        self.level = 0
        self.min_loud_chunks = int(loud_time_limit / sec_per_buffer)
        self.max_silence_duration = silence_time_limit
        self.silence_duration = 0
        self.silence_after_loud = silence_after_loud_time
        self.increase_multiplier = 200
        self.decrease_multiplier = 100

    def _increase_noise(self):
        if False:
            print('Hello World!')
        'Bumps the current level.\n\n        Modifies the noise level with a factor depending in the buffer length.\n        '
        if self.level < self.max_level:
            self.level += self.increase_multiplier * self.sec_per_buffer

    def _decrease_noise(self):
        if False:
            while True:
                i = 10
        'Decrease the current level.\n\n        Modifies the noise level with a factor depending in the buffer length.\n        '
        if self.level > self.min_level:
            self.level -= self.decrease_multiplier * self.sec_per_buffer

    def update(self, is_loud):
        if False:
            print('Hello World!')
        'Update the tracking. with either a loud chunk or a quiet chunk.\n\n        Args:\n            is_loud: True if a loud chunk should be registered\n                     False if a quiet chunk should be registered\n        '
        if is_loud:
            self._increase_noise()
            self.num_loud_chunks += 1
        else:
            self._decrease_noise()
        if self._quiet_enough():
            self.silence_duration += self.sec_per_buffer
        else:
            self.silence_duration = 0

    def _loud_enough(self):
        if False:
            while True:
                i = 10
        "Check if the noise loudness criteria is fulfilled.\n\n        The noise is considered loud enough if it's been over the threshold\n        for a certain number of chunks (accumulated, not in a row).\n        "
        return self.num_loud_chunks > self.min_loud_chunks

    def _quiet_enough(self):
        if False:
            i = 10
            return i + 15
        'Check if the noise quietness criteria is fulfilled.\n\n        The quiet level is instant and will return True if the level is lower\n        or equal to the minimum noise level.\n        '
        return self.level <= self.min_level

    def recording_complete(self):
        if False:
            return 10
        'Has the end creteria for the recording been met.\n\n        If the noise level has decresed from a loud level to a low level\n        the user has stopped speaking.\n\n        Alternatively if a lot of silence was recorded without detecting\n        a loud enough phrase.\n        '
        too_much_silence = self.silence_duration > self.max_silence_duration
        if too_much_silence:
            LOG.debug('Too much silence recorded without start of sentence detected')
        return (self._quiet_enough() and self.silence_duration > self.silence_after_loud) and (self._loud_enough() or too_much_silence)

class ResponsiveRecognizer(speech_recognition.Recognizer):
    SILENCE_SEC = 0.01
    MIN_LOUD_SEC_PER_PHRASE = 0.5
    MIN_SILENCE_AT_END = 0.25
    SEC_BETWEEN_WW_CHECKS = 0.2

    def __init__(self, wake_word_recognizer, watchdog=None):
        if False:
            print('Hello World!')
        self._watchdog = watchdog or (lambda : None)
        self.config = Configuration.get()
        listener_config = self.config.get('listener')
        self.upload_url = listener_config['wake_word_upload']['url']
        self.upload_disabled = listener_config['wake_word_upload']['disable']
        self.wake_word_name = wake_word_recognizer.key_phrase
        self.overflow_exc = listener_config.get('overflow_exception', False)
        super().__init__()
        self.wake_word_recognizer = wake_word_recognizer
        self.audio = pyaudio.PyAudio()
        self.multiplier = listener_config.get('multiplier')
        self.energy_ratio = listener_config.get('energy_ratio')
        self.save_utterances = listener_config.get('save_utterances', False)
        self.save_wake_words = listener_config.get('record_wake_words', False)
        self.save_path = listener_config.get('save_path', gettempdir())
        self.saved_wake_words_dir = join(self.save_path, 'mycroft_wake_words')
        if self.save_wake_words and (not isdir(self.saved_wake_words_dir)):
            os.mkdir(self.saved_wake_words_dir)
        self.saved_utterances_dir = join(self.save_path, 'mycroft_utterances')
        if self.save_utterances and (not isdir(self.saved_utterances_dir)):
            os.mkdir(self.saved_utterances_dir)
        self.mic_level_file = os.path.join(get_ipc_directory(), 'mic_level')
        self._stop_signaled = False
        self._listen_triggered = False
        self._account_id = None
        self.recording_timeout = listener_config.get('recording_timeout', 10.0)
        self.recording_timeout_with_silence = listener_config.get('recording_timeout_with_silence', 3.0)

    @property
    def account_id(self):
        if False:
            print('Hello World!')
        "Fetch account from backend when needed.\n\n        If an error occurs it's handled and a temporary value is returned.\n        When a value is received it will be cached until next start.\n        "
        if not self._account_id:
            try:
                self._account_id = DeviceApi().get()['user']['uuid']
            except (requests.RequestException, AttributeError):
                pass
            except Exception as e:
                LOG.debug('Unhandled exception while determining device_id, Error: {}'.format(repr(e)))
        return self._account_id or '0'

    def record_sound_chunk(self, source):
        if False:
            i = 10
            return i + 15
        return source.stream.read(source.CHUNK, self.overflow_exc)

    @staticmethod
    def calc_energy(sound_chunk, sample_width):
        if False:
            return 10
        return audioop.rms(sound_chunk, sample_width)

    def _record_phrase(self, source, sec_per_buffer, stream=None, ww_frames=None):
        if False:
            while True:
                i = 10
        "Record an entire spoken phrase.\n\n        Essentially, this code waits for a period of silence and then returns\n        the audio.  If silence isn't detected, it will terminate and return\n        a buffer of self.recording_timeout duration.\n\n        Args:\n            source (AudioSource):  Source producing the audio chunks\n            sec_per_buffer (float):  Fractional number of seconds in each chunk\n            stream (AudioStreamHandler): Stream target that will receive chunks\n                                         of the utterance audio while it is\n                                         being recorded.\n            ww_frames (deque):  Frames of audio data from the last part of wake\n                                word detection.\n\n        Returns:\n            bytearray: complete audio buffer recorded, including any\n                       silence at the end of the user's utterance\n        "
        noise_tracker = NoiseTracker(0, 25, sec_per_buffer, self.MIN_LOUD_SEC_PER_PHRASE, self.recording_timeout_with_silence)
        max_chunks = int(self.recording_timeout / sec_per_buffer)
        num_chunks = 0
        byte_data = get_silence(source.SAMPLE_WIDTH)
        if stream:
            stream.stream_start()
        phrase_complete = False
        while num_chunks < max_chunks and (not phrase_complete):
            if ww_frames:
                chunk = ww_frames.popleft()
            else:
                chunk = self.record_sound_chunk(source)
            byte_data += chunk
            num_chunks += 1
            if stream:
                stream.stream_chunk(chunk)
            energy = self.calc_energy(chunk, source.SAMPLE_WIDTH)
            test_threshold = self.energy_threshold * self.multiplier
            is_loud = energy > test_threshold
            noise_tracker.update(is_loud)
            if not is_loud:
                self._adjust_threshold(energy, sec_per_buffer)
            phrase_complete = noise_tracker.recording_complete() or check_for_signal('buttonPress')
            if num_chunks % 10 == 0:
                self._watchdog()
                self.write_mic_level(energy, source)
        return byte_data

    def write_mic_level(self, energy, source):
        if False:
            for i in range(10):
                print('nop')
        with open(self.mic_level_file, 'w') as f:
            f.write('Energy:  cur={} thresh={:.3f} muted={}'.format(energy, self.energy_threshold, int(source.muted)))

    def _skip_wake_word(self):
        if False:
            print('Hello World!')
        'Check if told programatically to skip the wake word\n\n        For example when we are in a dialog with the user.\n        '
        if self._listen_triggered:
            return True
        if check_for_signal('buttonPress', 1):
            sleep(0.25)
            if check_for_signal('buttonPress'):
                LOG.debug('Button Pressed, wakeword not needed')
                return True
        return False

    def stop(self):
        if False:
            return 10
        'Signal stop and exit waiting state.'
        self._stop_signaled = True

    def _compile_metadata(self):
        if False:
            i = 10
            return i + 15
        ww_module = self.wake_word_recognizer.__class__.__name__
        if ww_module == 'PreciseHotword':
            model_path = self.wake_word_recognizer.precise_model
            with open(model_path, 'rb') as f:
                model_hash = md5(f.read()).hexdigest()
        else:
            model_hash = '0'
        return {'name': self.wake_word_name.replace(' ', '-'), 'engine': md5(ww_module.encode('utf-8')).hexdigest(), 'time': str(int(1000 * get_time())), 'sessionId': SessionManager.get().session_id, 'accountId': self.account_id, 'model': str(model_hash)}

    def trigger_listen(self):
        if False:
            return 10
        'Externally trigger listening.'
        LOG.debug('Listen triggered from external source.')
        self._listen_triggered = True

    def _upload_wakeword(self, audio, metadata):
        if False:
            i = 10
            return i + 15
        'Upload the wakeword in a background thread.'
        LOG.debug('Wakeword uploading has been disabled. The API endpoint used in Mycroft-core v20.2 and below has been deprecated. To contribute new wakeword samples please upgrade to v20.8 or above.')

    def _send_wakeword_info(self, emitter):
        if False:
            while True:
                i = 10
        'Send messagebus message indicating that a wakeword was received.\n\n        Args:\n            emitter: bus emitter to send information on.\n        '
        SessionManager.touch()
        payload = {'utterance': self.wake_word_name, 'session': SessionManager.get().session_id}
        emitter.emit('recognizer_loop:wakeword', payload)

    def _write_wakeword_to_disk(self, audio, metadata):
        if False:
            for i in range(10):
                print('nop')
        'Write wakeword to disk.\n\n        Args:\n            audio: Audio data to write\n            metadata: List of metadata about the captured wakeword\n        '
        filename = join(self.saved_wake_words_dir, '_'.join((str(metadata[k]) for k in sorted(metadata))) + '.wav')
        with open(filename, 'wb') as f:
            f.write(audio.get_wav_data())

    def _handle_wakeword_found(self, audio_data, source):
        if False:
            return 10
        'Perform actions to be triggered after a wakeword is found.\n\n        This includes: emit event on messagebus that a wakeword is heard,\n        store wakeword to disk if configured and sending the wakeword data\n        to the cloud in case the user has opted into the data sharing.\n        '
        upload_allowed = self.config['opt_in'] and (not self.upload_disabled)
        if self.save_wake_words or upload_allowed:
            audio = self._create_audio_data(audio_data, source)
            metadata = self._compile_metadata()
            if self.save_wake_words:
                self._write_wakeword_to_disk(audio, metadata)
            if upload_allowed:
                self._upload_wakeword(audio, metadata)

    def _wait_until_wake_word(self, source, sec_per_buffer):
        if False:
            while True:
                i = 10
        'Listen continuously on source until a wake word is spoken\n\n        Args:\n            source (AudioSource):  Source producing the audio chunks\n            sec_per_buffer (float):  Fractional number of seconds in each chunk\n        '
        ww_duration = self.wake_word_recognizer.expected_duration
        ww_test_duration = max(3, ww_duration)
        mic_write_counter = 0
        num_silent_bytes = int(self.SILENCE_SEC * source.SAMPLE_RATE * source.SAMPLE_WIDTH)
        silence = get_silence(num_silent_bytes)
        max_size = source.duration_to_bytes(ww_duration)
        test_size = source.duration_to_bytes(ww_test_duration)
        audio_buffer = CyclicAudioBuffer(max_size, silence)
        buffers_per_check = self.SEC_BETWEEN_WW_CHECKS / sec_per_buffer
        buffers_since_check = 0.0
        average_samples = int(5 / sec_per_buffer)
        audio_mean = RollingMean(average_samples)
        ww_frames = deque(maxlen=7)
        said_wake_word = False
        audio_data = None
        while not said_wake_word and (not self._stop_signaled) and (not self._skip_wake_word()):
            chunk = self.record_sound_chunk(source)
            audio_buffer.append(chunk)
            ww_frames.append(chunk)
            energy = self.calc_energy(chunk, source.SAMPLE_WIDTH)
            audio_mean.append_sample(energy)
            if energy < self.energy_threshold * self.multiplier:
                self._adjust_threshold(energy, sec_per_buffer)
            if self.energy_threshold < energy < audio_mean.value * 1.5:
                self.energy_threshold = energy * 1.2
            if mic_write_counter % 3:
                self._watchdog()
                self.write_mic_level(energy, source)
            mic_write_counter += 1
            buffers_since_check += 1.0
            self.wake_word_recognizer.update(chunk)
            if buffers_since_check > buffers_per_check:
                buffers_since_check -= buffers_per_check
                audio_data = audio_buffer.get_last(test_size) + silence
                said_wake_word = self.wake_word_recognizer.found_wake_word(audio_data)
        self._listen_triggered = False
        return WakeWordData(audio_data, said_wake_word, self._stop_signaled, ww_frames)

    @staticmethod
    def _create_audio_data(raw_data, source):
        if False:
            return 10
        '\n        Constructs an AudioData instance with the same parameters\n        as the source and the specified frame_data\n        '
        return AudioData(raw_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

    def mute_and_confirm_listening(self, source):
        if False:
            return 10
        audio_file = resolve_resource_file(self.config.get('sounds').get('start_listening'))
        if audio_file:
            source.mute()
            play_wav(audio_file).wait()
            source.unmute()
            return True
        else:
            return False

    def listen(self, source, emitter, stream=None):
        if False:
            while True:
                i = 10
        "Listens for chunks of audio that Mycroft should perform STT on.\n\n        This will listen continuously for a wake-up-word, then return the\n        audio chunk containing the spoken phrase that comes immediately\n        afterwards.\n\n        Args:\n            source (AudioSource):  Source producing the audio chunks\n            emitter (EventEmitter): Emitter for notifications of when recording\n                                    begins and ends.\n            stream (AudioStreamHandler): Stream target that will receive chunks\n                                         of the utterance audio while it is\n                                         being recorded\n\n        Returns:\n            AudioData: audio with the user's utterance, minus the wake-up-word\n        "
        assert isinstance(source, AudioSource), 'Source must be an AudioSource'
        sec_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE
        self.adjust_for_ambient_noise(source, 1.0)
        LOG.debug('Waiting for wake word...')
        ww_data = self._wait_until_wake_word(source, sec_per_buffer)
        ww_frames = None
        if ww_data.found:
            self._send_wakeword_info(emitter)
            self._handle_wakeword_found(ww_data.audio, source)
            ww_frames = ww_data.end_audio
        if ww_data.stopped:
            return
        LOG.debug('Recording...')
        if self.config.get('confirm_listening'):
            if self.mute_and_confirm_listening(source):
                ww_frames = None
        emitter.emit('recognizer_loop:record_begin')
        frame_data = self._record_phrase(source, sec_per_buffer, stream, ww_frames)
        audio_data = self._create_audio_data(frame_data, source)
        emitter.emit('recognizer_loop:record_end')
        if self.save_utterances:
            LOG.info('Recording utterance')
            stamp = str(datetime.datetime.now())
            filename = '/{}/{}.wav'.format(self.saved_utterances_dir, stamp)
            with open(filename, 'wb') as filea:
                filea.write(audio_data.get_wav_data())
            LOG.debug('Thinking...')
        return audio_data

    def _adjust_threshold(self, energy, seconds_per_buffer):
        if False:
            for i in range(10):
                print('nop')
        if self.dynamic_energy_threshold and energy > 0:
            damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer
            target_energy = energy * self.energy_ratio
            self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)