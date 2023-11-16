import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def transcribe_streaming_voice_activity_events(project_id: str, audio_file: str) -> cloud_speech.StreamingRecognizeResponse:
    if False:
        i = 10
        return i + 15
    'Transcribes audio from a file into text.\n\n    Args:\n        project_id: The GCP project ID to use.\n        audio_file: The path to the audio file to transcribe.\n\n    Returns:\n        The streaming response containing the transcript.\n    '
    client = SpeechClient()
    with open(audio_file, 'rb') as f:
        content = f.read()
    chunk_length = len(content) // 5
    stream = [content[start:start + chunk_length] for start in range(0, len(content), chunk_length)]
    audio_requests = (cloud_speech.StreamingRecognizeRequest(audio=audio) for audio in stream)
    recognition_config = cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), language_codes=['en-US'], model='long')
    streaming_features = cloud_speech.StreamingRecognitionFeatures(enable_voice_activity_events=True)
    streaming_config = cloud_speech.StreamingRecognitionConfig(config=recognition_config, streaming_features=streaming_features)
    config_request = cloud_speech.StreamingRecognizeRequest(recognizer=f'projects/{project_id}/locations/global/recognizers/_', streaming_config=streaming_config)

    def requests(config: cloud_speech.RecognitionConfig, audio: list) -> list:
        if False:
            print('Hello World!')
        yield config
        yield from audio
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
    parser.add_argument('audio_file', help='Audio file to stream')
    args = parser.parse_args()
    transcribe_streaming_voice_activity_events(args.project_id, args.audio_file)