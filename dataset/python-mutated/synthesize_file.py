"""Google Cloud Text-To-Speech API sample application .

Example usage:
    python synthesize_file.py --text resources/hello.txt
    python synthesize_file.py --ssml resources/hello.ssml
"""
import argparse

def synthesize_text_file(text_file):
    if False:
        return 10
    'Synthesizes speech from the input file of text.'
    from google.cloud import texttospeech
    client = texttospeech.TextToSpeechClient()
    with open(text_file) as f:
        text = f.read()
        input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code='en-US', ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(request={'input': input_text, 'voice': voice, 'audio_config': audio_config})
    with open('output.mp3', 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

def synthesize_ssml_file(ssml_file):
    if False:
        print('Hello World!')
    'Synthesizes speech from the input file of ssml.\n\n    Note: ssml must be well-formed according to:\n        https://www.w3.org/TR/speech-synthesis/\n    '
    from google.cloud import texttospeech
    client = texttospeech.TextToSpeechClient()
    with open(ssml_file) as f:
        ssml = f.read()
        input_text = texttospeech.SynthesisInput(ssml=ssml)
    voice = texttospeech.VoiceSelectionParams(language_code='en-US', ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    with open('output.mp3', 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', help='The text file from which to synthesize speech.')
    group.add_argument('--ssml', help='The ssml file from which to synthesize speech.')
    args = parser.parse_args()
    if args.text:
        synthesize_text_file(args.text)
    else:
        synthesize_ssml_file(args.ssml)