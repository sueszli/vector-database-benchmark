import argparse
import io
from google.cloud import speech_v1p1beta1
import grpc

def transcribe_onprem(local_file_path: str, api_endpoint: str) -> speech_v1p1beta1.RecognizeResponse:
    if False:
        return 10
    '\n    Transcribe a short audio file using synchronous speech recognition on-prem\n\n    Args:\n      local_file_path: The path to local audio file, e.g. /path/audio.wav\n      api_endpoint: Endpoint to call for speech recognition, e.g. 0.0.0.0:10000\n\n    Returns:\n      The speech recognition response\n          {\n    '
    channel = grpc.insecure_channel(target=api_endpoint)
    transport = speech_v1p1beta1.services.speech.transports.SpeechGrpcTransport(channel=channel)
    client = speech_v1p1beta1.SpeechClient(transport=transport)
    language_code = 'en-US'
    sample_rate_hertz = 16000
    encoding = speech_v1p1beta1.RecognitionConfig.AudioEncoding.LINEAR16
    config = {'encoding': encoding, 'language_code': language_code, 'sample_rate_hertz': sample_rate_hertz}
    with io.open(local_file_path, 'rb') as f:
        content = f.read()
    audio = {'content': content}
    response = client.recognize(request={'config': config, 'audio': audio})
    for result in response.results:
        alternative = result.alternatives[0]
        print(f'Transcript: {alternative.transcript}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--file_path', required=True, help='Path to local audio file to be recognized, e.g. /path/audio.wav')
    parser.add_argument('--api_endpoint', required=True, help='Endpoint to call for speech recognition, e.g. 0.0.0.0:10000')
    args = parser.parse_args()
    transcribe_onprem(local_file_path=args.file_path, api_endpoint=args.api_endpoint)