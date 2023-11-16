from google.cloud import speech_v1p1beta1 as speech

def sample_recognize(storage_uri: str) -> speech.RecognizeResponse:
    if False:
        print('Hello World!')
    '\n    Performs synchronous speech recognition on an audio file\n\n    Args:\n      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]\n    '
    client = speech.SpeechClient()
    language_code = 'en-US'
    sample_rate_hertz = 44100
    encoding = speech.RecognitionConfig.AudioEncoding.MP3
    config = {'language_code': language_code, 'sample_rate_hertz': sample_rate_hertz, 'encoding': encoding}
    audio = {'uri': storage_uri}
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        alternative = result.alternatives[0]
        print(f'Transcript: {alternative.transcript}')
    return response

def main() -> None:
    if False:
        i = 10
        return i + 15
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_uri', type=str, default='gs://cloud-samples-data/speech/brooklyn_bridge.mp3')
    args = parser.parse_args()
    sample_recognize(args.storage_uri)
if __name__ == '__main__':
    main()