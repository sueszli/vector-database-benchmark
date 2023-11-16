import html
from google.cloud import texttospeech

def ssml_to_audio(ssml_text, outfile):
    if False:
        return 10
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
    voice = texttospeech.VoiceSelectionParams(language_code='en-US', ssml_gender=texttospeech.SsmlVoiceGender.MALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open(outfile, 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file ' + outfile)

def text_to_ssml(inputfile):
    if False:
        i = 10
        return i + 15
    with open(inputfile) as f:
        raw_lines = f.read()
    escaped_lines = html.escape(raw_lines)
    ssml = '<speak>{}</speak>'.format(escaped_lines.replace('\n', '\n<break time="2s"/>'))
    return ssml

def main():
    if False:
        print('Hello World!')
    plaintext = 'resources/example.txt'
    ssml_text = text_to_ssml(plaintext)
    ssml_to_audio(ssml_text, 'resources/example.mp3')
if __name__ == '__main__':
    main()