"""Google Cloud Text-To-Speech API sample application for audio profile.

Example usage:
    python audio_profile.py --text "hello" --effects_profile_id
        "telephony-class-application" --output "output.mp3"
"""
import argparse

def synthesize_text_with_audio_profile(text, output, effects_profile_id):
    if False:
        return 10
    'Synthesizes speech from the input string of text.'
    from google.cloud import texttospeech
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code='en-US')
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, effects_profile_id=[effects_profile_id])
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    with open(output, 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file "%s"' % output)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', help='The output mp3 file.')
    parser.add_argument('--text', help='The text from which to synthesize speech.')
    parser.add_argument('--effects_profile_id', help='The audio effects profile id to be applied.')
    args = parser.parse_args()
    synthesize_text_with_audio_profile(args.text, args.output, args.effects_profile_id)