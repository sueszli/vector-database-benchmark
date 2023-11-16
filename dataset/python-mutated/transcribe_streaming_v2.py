import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def transcribe_streaming_v2(project_id: str, audio_file: str) -> cloud_speech.StreamingRecognizeResponse:
    if False:
        print('Hello World!')
    'Transcribes audio from audio file stream.\n\n    Args:\n        project_id: The GCP project ID.\n        audio_file: The path to the audio file to transcribe.\n\n    Returns:\n        The response from the transcribe method.\n    '
    client = SpeechClient()
    with open(audio_file, 'rb') as f:
        content = f.read()
    chunk_length = len(content) // 5
    stream = [content[start:start + chunk_length] for start in range(0, len(content), chunk_length)]
    audio_requests = (cloud_speech.StreamingRecognizeRequest(audio=audio) for audio in stream)
    recognition_config = cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), language_codes=['en-US'], model='long')
    streaming_config = cloud_speech.StreamingRecognitionConfig(config=recognition_config)
    config_request = cloud_speech.StreamingRecognizeRequest(recognizer=f'projects/{project_id}/locations/global/recognizers/_', streaming_config=streaming_config)

    def requests(config: cloud_speech.RecognitionConfig, audio: list) -> list:
        if False:
            for i in range(10):
                print('nop')
        yield config
        yield from audio
    responses_iterator = client.streaming_recognize(requests=requests(config_request, audio_requests))
    responses = []
    for response in responses_iterator:
        responses.append(response)
        for result in response.results:
            print(f'Transcript: {result.alternatives[0].transcript}')
    return responses
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('audio_file', help='Audio file to stream')
    args = parser.parse_args()
    transcribe_streaming_v2(args.project_id, args.audio_file)