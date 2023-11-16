"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the additional dependency `pyaudio`. To install
using pip:

    pip install pyaudio

Example usage:
    python transcribe_streaming_mic.py
"""
import queue
import re
import sys
from google.cloud import speech
import pyaudio
RATE = 16000
CHUNK = int(RATE / 10)

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self: object, rate: int=RATE, chunk: int=CHUNK) -> None:
        if False:
            i = 10
            return i + 15
        'The audio -- and generator -- is guaranteed to be on the main thread.'
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self: object) -> object:
        if False:
            print('Hello World!')
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(format=pyaudio.paInt16, channels=1, rate=self._rate, input=True, frames_per_buffer=self._chunk, stream_callback=self._fill_buffer)
        self.closed = False
        return self

    def __exit__(self: object, type: object, value: object, traceback: object) -> None:
        if False:
            return 10
        'Closes the stream, regardless of whether the connection was lost or not.'
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self: object, in_data: object, frame_count: int, time_info: object, status_flags: object) -> object:
        if False:
            while True:
                i = 10
        'Continuously collect data from the audio stream, into the buffer.\n\n        Args:\n            in_data: The audio data as a bytes object\n            frame_count: The number of frames captured\n            time_info: The time information\n            status_flags: The status flags\n\n        Returns:\n            The audio data as a bytes object\n        '
        self._buff.put(in_data)
        return (None, pyaudio.paContinue)

    def generator(self: object) -> object:
        if False:
            while True:
                i = 10
        'Generates audio chunks from the stream of audio data in chunks.\n\n        Args:\n            self: The MicrophoneStream object\n\n        Returns:\n            A generator that outputs audio chunks.\n        '
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b''.join(data)

def listen_print_loop(responses: object) -> str:
    if False:
        i = 10
        return i + 15
    'Iterates through server responses and prints them.\n\n    The responses passed is a generator that will block until a response\n    is provided by the server.\n\n    Each response may contain multiple results, and each result may contain\n    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we\n    print only the transcription for the top alternative of the top result.\n\n    In this case, responses are provided for interim results as well. If the\n    response is an interim one, print a line feed at the end of it, to allow\n    the next result to overwrite it, until the response is a final one. For the\n    final one, print a newline to preserve the finalized transcription.\n\n    Args:\n        responses: List of server responses\n\n    Returns:\n        The transcribed text.\n    '
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))
        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            print(transcript + overwrite_chars)
            if re.search('\\b(exit|quit)\\b', transcript, re.I):
                print('Exiting..')
                break
            num_chars_printed = 0
        return transcript

def main() -> None:
    if False:
        while True:
            i = 10
    'Transcribe speech from audio file.'
    language_code = 'en-US'
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=RATE, language_code=language_code)
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)
        listen_print_loop(responses)
if __name__ == '__main__':
    main()