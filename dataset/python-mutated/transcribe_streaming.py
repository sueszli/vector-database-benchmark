"""Google Cloud Speech API sample application using the streaming API.

Example usage:
    python transcribe_streaming.py resources/audio.raw
"""
import argparse
from google.cloud import speech

def transcribe_streaming(stream_file: str) -> speech.RecognitionConfig:
    if False:
        i = 10
        return i + 15
    'Streams transcription of the given audio file.'
    client = speech.SpeechClient()
    with open(stream_file, 'rb') as audio_file:
        content = audio_file.read()
    stream = [content]
    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code='en-US')
    streaming_config = speech.StreamingRecognitionConfig(config=config)
    responses = client.streaming_recognize(config=streaming_config, requests=requests)
    for response in responses:
        for result in response.results:
            print(f'Finished: {result.is_final}')
            print(f'Stability: {result.stability}')
            alternatives = result.alternatives
            for alternative in alternatives:
                print(f'Confidence: {alternative.confidence}')
                print(f'Transcript: {alternative.transcript}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stream', help='File to stream to the API')
    args = parser.parse_args()
    transcribe_streaming(args.stream)