"""Google Cloud Text-To-Speech API sample application .

Example usage:
    python synthesize_text.py --text "hello"
    python synthesize_text.py --ssml "<speak>Hello there.</speak>"
"""
import argparse

def synthesize_text(text):
    if False:
        i = 10
        return i + 15
    'Synthesizes speech from the input string of text.'
    from google.cloud import texttospeech
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code='en-US', name='en-US-Standard-C', ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(request={'input': input_text, 'voice': voice, 'audio_config': audio_config})
    with open('output.mp3', 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

def synthesize_ssml(ssml):
    if False:
        while True:
            i = 10
    'Synthesizes speech from the input string of ssml.\n\n    Note: ssml must be well-formed according to:\n        https://www.w3.org/TR/speech-synthesis/\n\n    Example: <speak>Hello there.</speak>\n    '
    from google.cloud import texttospeech
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(ssml=ssml)
    voice = texttospeech.VoiceSelectionParams(language_code='en-US', name='en-US-Standard-C', ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    with open('output.mp3', 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', help='The text from which to synthesize speech.')
    group.add_argument('--ssml', help='The ssml string from which to synthesize speech.')
    args = parser.parse_args()
    if args.text:
        synthesize_text(args.text)
    else:
        synthesize_ssml(args.ssml)