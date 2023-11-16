"""Google Cloud Dialogflow API sample code using the StreamingAnalyzeContent
API.

Also please contact Google to get credentials of this project and set up the
credential file json locations by running:
export GOOGLE_APPLICATION_CREDENTIALS=<cred_json_file_location>

Example usage:
    export GOOGLE_CLOUD_PROJECT='cloud-contact-center-ext-demo'
    export CONVERSATION_PROFILE='FnuBYO8eTBWM8ep1i-eOng'
    export GOOGLE_APPLICATION_CREDENTIALS='/Users/ruogu/Desktop/keys/cloud-contact-center-ext-demo-78798f9f9254.json'
    python streaming_transcription.py

Then started to talk in English, you should see transcription shows up as you speak.

Say "Quit" or "Exit" to stop.
"""
import os
import re
import sys
from google.api_core.exceptions import DeadlineExceeded
import pyaudio
from six.moves import queue
import conversation_management
import participant_management
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
CONVERSATION_PROFILE_ID = os.getenv('CONVERSATION_PROFILE')
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)
RESTART_TIMEOUT = 160
MAX_LOOKBACK = 3
YELLOW = '\x1b[0;33m'

class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size):
        if False:
            i = 10
            return i + 15
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.is_final = False
        self.closed = True
        self.restart_counter = 0
        self.last_start_time = 0
        self.is_final_offset = 0
        self.audio_input_chunks = []
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(format=pyaudio.paInt16, channels=self._num_channels, rate=self._rate, input=True, frames_per_buffer=self.chunk_size, stream_callback=self._fill_buffer)

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Continuously collect data from the audio stream, into the buffer in\n        chunksize.'
        self._buff.put(in_data)
        return (None, pyaudio.paContinue)

    def generator(self):
        if False:
            print('Hello World!')
        'Stream Audio from microphone to API and to local buffer'
        try:
            print('restart generator')
            self.is_final = False
            total_processed_time = self.last_start_time + self.is_final_offset
            processed_bytes_length = int(total_processed_time * SAMPLE_RATE * 16 / 8) / 1000
            self.last_start_time = total_processed_time
            if processed_bytes_length != 0:
                audio_bytes = b''.join(self.audio_input_chunks)
                need_to_process_length = min(int(len(audio_bytes) - processed_bytes_length), int(MAX_LOOKBACK * SAMPLE_RATE * 16 / 8))
                need_to_process_bytes = audio_bytes[-1 * need_to_process_length:]
                yield need_to_process_bytes
            while not self.closed and (not self.is_final):
                data = []
                chunk = self._buff.get()
                if chunk is None:
                    return
                data.append(chunk)
                while True:
                    try:
                        chunk = self._buff.get(block=False)
                        if chunk is None:
                            return
                        data.append(chunk)
                    except queue.Empty:
                        break
                self.audio_input_chunks.extend(data)
                if data:
                    yield b''.join(data)
        finally:
            print('Stop generator')

def main():
    if False:
        for i in range(10):
            print('nop')
    'start bidirectional streaming from microphone input to Dialogflow API'
    conversation = conversation_management.create_conversation(project_id=PROJECT_ID, conversation_profile_id=CONVERSATION_PROFILE_ID)
    conversation_id = conversation.name.split('conversations/')[1].rstrip()
    end_user = participant_management.create_participant(project_id=PROJECT_ID, conversation_id=conversation_id, role='END_USER')
    participant_id = end_user.name.split('participants/')[1].rstrip()
    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    print(mic_manager.chunk_size)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write('End (ms)       Transcript Results/Status\n')
    sys.stdout.write('=====================================================\n')
    with mic_manager as stream:
        while not stream.closed:
            terminate = False
            while not terminate:
                try:
                    print(f'New Streaming Analyze Request: {stream.restart_counter}')
                    stream.restart_counter += 1
                    responses = participant_management.analyze_content_audio_stream(conversation_id=conversation_id, participant_id=participant_id, sample_rate_herz=SAMPLE_RATE, stream=stream, timeout=RESTART_TIMEOUT, language_code='en-US', single_utterance=False)
                    for response in responses:
                        if response.message:
                            print(response)
                        if response.recognition_result.is_final:
                            print(response)
                            offset = response.recognition_result.speech_end_offset
                            stream.is_final_offset = int(offset.seconds * 1000 + offset.microseconds / 1000)
                            transcript = response.recognition_result.transcript
                            stream.is_final = True
                            if re.search('\\b(exit|quit)\\b', transcript, re.I):
                                sys.stdout.write(YELLOW)
                                sys.stdout.write('Exiting...\n')
                                terminate = True
                                stream.closed = True
                                break
                except DeadlineExceeded:
                    print('Deadline Exceeded, restarting.')
            if terminate:
                conversation_management.complete_conversation(project_id=PROJECT_ID, conversation_id=conversation_id)
                break
if __name__ == '__main__':
    main()