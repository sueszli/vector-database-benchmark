import argparse
from time import sleep
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.protobuf import duration_pb2

def transcribe_streaming_voice_activity_timeouts(project_id: str, speech_start_timeout: int, speech_end_timeout: int, audio_file: str) -> cloud_speech.StreamingRecognizeResponse:
    if False:
        return 10
    'Transcribes audio from audio file to text.\n\n    Args:\n        project_id: The GCP project ID to use.\n        speech_start_timeout: The timeout in seconds for speech start.\n        speech_end_timeout: The timeout in seconds for speech end.\n        audio_file: The audio file to transcribe.\n\n    Returns:\n        The streaming response containing the transcript.\n    '
    client = SpeechClient()
    with open(audio_file, 'rb') as f:
        content = f.read()
    chunk_length = len(content) // 20
    stream = [content[start:start + chunk_length] for start in range(0, len(content), chunk_length)]
    audio_requests = (cloud_speech.StreamingRecognizeRequest(audio=audio) for audio in stream)
    recognition_config = cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), language_codes=['en-US'], model='long')
    speech_start_timeout = duration_pb2.Duration(seconds=speech_start_timeout)
    speech_end_timeout = duration_pb2.Duration(seconds=speech_end_timeout)
    voice_activity_timeout = cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(speech_start_timeout=speech_start_timeout, speech_end_timeout=speech_end_timeout)
    streaming_features = cloud_speech.StreamingRecognitionFeatures(enable_voice_activity_events=True, voice_activity_timeout=voice_activity_timeout)
    streaming_config = cloud_speech.StreamingRecognitionConfig(config=recognition_config, streaming_features=streaming_features)
    config_request = cloud_speech.StreamingRecognizeRequest(recognizer=f'projects/{project_id}/locations/global/recognizers/_', streaming_config=streaming_config)

    def requests(config: cloud_speech.RecognitionConfig, audio: list) -> list:
        if False:
            i = 10
            return i + 15
        yield config
        for message in audio:
            sleep(0.5)
            yield message
    responses_iterator = client.streaming_recognize(requests=requests(config_request, audio_requests))
    responses = []
    for response in responses_iterator:
        responses.append(response)
        if response.speech_event_type == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN:
            print('Speech started.')
        if response.speech_event_type == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END:
            print('Speech ended.')
        for result in response.results:
            print(f'Transcript: {result.alternatives[0].transcript}')
    return responses
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('speech_start_timeout', help='Timeout in seconds for speech start')
    parser.add_argument('speech_end_timeout', help='Timeout in seconds for speech end')
    parser.add_argument('audio_file', help='Audio file to stream')
    args = parser.parse_args()
    transcribe_streaming_voice_activity_timeouts(args.project_id, args.speech_start_timeout, args.speech_end_timeout, args.audio_file)