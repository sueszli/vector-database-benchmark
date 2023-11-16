"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the dependencies `pyaudio` and `termcolor`.
To install using pip:

    pip install pyaudio
    pip install termcolor

Example usage:
    python transcribe_streaming_infinite.py
"""
import queue
import re
import sys
import time
from google.cloud import speech
import pyaudio
STREAMING_LIMIT = 240000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)
RED = '\x1b[0;31m'
GREEN = '\x1b[0;32m'
YELLOW = '\x1b[0;33m'

def get_current_time() -> int:
    if False:
        print('Hello World!')
    'Return Current Time in MS.\n\n    Returns:\n        int: Current Time in MS.\n    '
    return int(round(time.time() * 1000))

class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self: object, rate: int, chunk_size: int) -> None:
        if False:
            return 10
        "Creates a resumable microphone stream.\n\n        Args:\n        self: The class instance.\n        rate: The audio file's sampling rate.\n        chunk_size: The audio file's chunk size.\n\n        returns: None\n        "
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(format=pyaudio.paInt16, channels=self._num_channels, rate=self._rate, input=True, frames_per_buffer=self.chunk_size, stream_callback=self._fill_buffer)

    def __enter__(self: object) -> object:
        if False:
            i = 10
            return i + 15
        'Opens the stream.\n\n        Args:\n        self: The class instance.\n\n        returns: None\n        '
        self.closed = False
        return self

    def __exit__(self: object, type: object, value: object, traceback: object) -> object:
        if False:
            i = 10
            return i + 15
        'Closes the stream and releases resources.\n\n        Args:\n        self: The class instance.\n        type: The exception type.\n        value: The exception value.\n        traceback: The exception traceback.\n\n        returns: None\n        '
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self: object, in_data: object, *args: object, **kwargs: object) -> object:
        if False:
            print('Hello World!')
        'Continuously collect data from the audio stream, into the buffer.\n\n        Args:\n        self: The class instance.\n        in_data: The audio data as a bytes object.\n        args: Additional arguments.\n        kwargs: Additional arguments.\n\n        returns: None\n        '
        self._buff.put(in_data)
        return (None, pyaudio.paContinue)

    def generator(self: object) -> object:
        if False:
            print('Hello World!')
        'Stream Audio from microphone to API and to local buffer\n\n        Args:\n            self: The class instance.\n\n        returns:\n            The data from the audio stream.\n        '
        while not self.closed:
            data = []
            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)
                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0
                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time
                    chunks_from_ms = round((self.final_request_end_time - self.bridging_offset) / chunk_time)
                    self.bridging_offset = round((len(self.last_audio_input) - chunks_from_ms) * chunk_time)
                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])
                self.new_stream = False
            chunk = self._buff.get()
            self.audio_input.append(chunk)
            if chunk is None:
                return
            data.append(chunk)
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)
                except queue.Empty:
                    break
            yield b''.join(data)

def listen_print_loop(responses: object, stream: object) -> object:
    if False:
        i = 10
        return i + 15
    'Iterates through server responses and prints them.\n\n    The responses passed is a generator that will block until a response\n    is provided by the server.\n\n    Each response may contain multiple results, and each result may contain\n    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we\n    print only the transcription for the top alternative of the top result.\n\n    In this case, responses are provided for interim results as well. If the\n    response is an interim one, print a line feed at the end of it, to allow\n    the next result to overwrite it, until the response is a final one. For the\n    final one, print a newline to preserve the finalized transcription.\n\n    Arg:\n        responses: The responses returned from the API.\n        stream: The audio stream to be processed.\n\n    Returns:\n        The transcript of the result\n    '
    for response in responses:
        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        result_seconds = 0
        result_micros = 0
        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds
        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds
        stream.result_end_time = int(result_seconds * 1000 + result_micros / 1000)
        corrected_time = stream.result_end_time - stream.bridging_offset + STREAMING_LIMIT * stream.restart_counter
        if result.is_final:
            sys.stdout.write(GREEN)
            sys.stdout.write('\x1b[K')
            sys.stdout.write(str(corrected_time) + ': ' + transcript + '\n')
            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True
            if re.search('\\b(exit|quit)\\b', transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write('Exiting...\n')
                stream.closed = True
                break
        else:
            sys.stdout.write(RED)
            sys.stdout.write('\x1b[K')
            sys.stdout.write(str(corrected_time) + ': ' + transcript + '\r')
            stream.last_transcript_was_final = False
        return transcript

def main() -> None:
    if False:
        print('Hello World!')
    'start bidirectional streaming from microphone input to speech API'
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=SAMPLE_RATE, language_code='en-US', max_alternatives=1)
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)
    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    print(mic_manager.chunk_size)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write('End (ms)       Transcript Results/Status\n')
    sys.stdout.write('=====================================================\n')
    with mic_manager as stream:
        while not stream.closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write('\n' + str(STREAMING_LIMIT * stream.restart_counter) + ': NEW REQUEST\n')
            stream.audio_input = []
            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
            responses = client.streaming_recognize(streaming_config, requests)
            listen_print_loop(responses, stream)
            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1
            if not stream.last_transcript_was_final:
                sys.stdout.write('\n')
            stream.new_stream = True
if __name__ == '__main__':
    main()