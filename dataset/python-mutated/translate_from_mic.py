"""Cloud Media Translation API sample application using a microphone.

Example usage:
    python translate_from_mic.py
"""
import itertools
import queue
from google.cloud import mediatranslation as media
import pyaudio
RATE = 16000
CHUNK = int(RATE / 10)
SpeechEventType = media.StreamingTranslateSpeechResponse.SpeechEventType

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        if False:
            i = 10
            return i + 15
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(format=pyaudio.paInt16, channels=1, rate=self._rate, input=True, frames_per_buffer=self._chunk, stream_callback=self._fill_buffer)
        self.closed = False
        return self

    def __exit__(self, type=None, value=None, traceback=None):
        if False:
            print('Hello World!')
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        if False:
            i = 10
            return i + 15
        'Continuously collect data from the audio stream, into the buffer.'
        self._buff.put(in_data)
        return (None, pyaudio.paContinue)

    def exit(self):
        if False:
            i = 10
            return i + 15
        self.__exit__()

    def generator(self):
        if False:
            i = 10
            return i + 15
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

def listen_print_loop(responses):
    if False:
        i = 10
        return i + 15
    'Iterates through server responses and prints them.\n\n    The responses passed is a generator that will block until a response\n    is provided by the server.\n    '
    translation = ''
    for response in responses:
        if response.speech_event_type == SpeechEventType.END_OF_SINGLE_UTTERANCE:
            print(f'\nFinal translation: {translation}')
            return 0
        result = response.result
        translation = result.text_translation_result.translation
        print(f'\nPartial translation: {translation}')

def do_translation_loop():
    if False:
        while True:
            i = 10
    print('Begin speaking...')
    client = media.SpeechTranslationServiceClient()
    speech_config = media.TranslateSpeechConfig(audio_encoding='linear16', source_language_code='en-US', target_language_code='es-ES')
    config = media.StreamingTranslateSpeechConfig(audio_config=speech_config, single_utterance=True)
    first_request = media.StreamingTranslateSpeechRequest(streaming_config=config)
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        mic_requests = (media.StreamingTranslateSpeechRequest(audio_content=content) for content in audio_generator)
        requests = itertools.chain(iter([first_request]), mic_requests)
        responses = client.streaming_translate_speech(requests)
        result = listen_print_loop(responses)
        if result == 0:
            stream.exit()

def main():
    if False:
        return 10
    while True:
        print()
        option = input("Press any key to translate or 'q' to quit: ")
        if option.lower() == 'q':
            break
        do_translation_loop()
if __name__ == '__main__':
    main()