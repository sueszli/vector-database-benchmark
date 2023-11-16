from google.cloud import speech_v1p1beta1 as speech

def sample_recognize(storage_uri: str, phrase: str) -> speech.RecognizeResponse:
    if False:
        for i in range(10):
            print('nop')
    '\n    Transcribe a short audio file with speech adaptation.\n\n    Args:\n      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]\n      phrase Phrase "hints" help recognize the specified phrases from your audio.\n    '
    client = speech.SpeechClient()
    phrases = [phrase]
    boost = 20.0
    speech_contexts_element = {'phrases': phrases, 'boost': boost}
    speech_contexts = [speech_contexts_element]
    sample_rate_hertz = 44100
    language_code = 'en-US'
    encoding = speech.RecognitionConfig.AudioEncoding.MP3
    config = {'speech_contexts': speech_contexts, 'sample_rate_hertz': sample_rate_hertz, 'language_code': language_code, 'encoding': encoding}
    audio = {'uri': storage_uri}
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        alternative = result.alternatives[0]
        print(f'Transcript: {alternative.transcript}')
    return response

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_uri', type=str, default='gs://cloud-samples-data/speech/brooklyn_bridge.mp3')
    parser.add_argument('--phrase', type=str, default='Brooklyn Bridge')
    args = parser.parse_args()
    sample_recognize(args.storage_uri, args.phrase)
if __name__ == '__main__':
    main()