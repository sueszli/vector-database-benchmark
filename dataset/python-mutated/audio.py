from pathlib import Path
from openai import OpenAI
openai = OpenAI()
speech_file_path = Path(__file__).parent / 'speech.mp3'

def main() -> None:
    if False:
        return 10
    response = openai.audio.speech.create(model='tts-1', voice='alloy', input='the quick brown fox jumped over the lazy dogs')
    response.stream_to_file(speech_file_path)
    transcription = openai.audio.transcriptions.create(model='whisper-1', file=speech_file_path)
    print(transcription.text)
    translation = openai.audio.translations.create(model='whisper-1', file=speech_file_path)
    print(translation.text)
if __name__ == '__main__':
    main()