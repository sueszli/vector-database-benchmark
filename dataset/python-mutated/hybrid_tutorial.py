import html
import os
from google.api_core.exceptions import AlreadyExists
from google.cloud import texttospeech
from google.cloud import translate_v3beta1 as translate
from google.cloud import vision
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def pic_to_text(infile: str) -> str:
    if False:
        return 10
    'Detects text in an image file\n\n    Args:\n    infile: path to image file\n\n    Returns:\n    String of text detected in image\n    '
    client = vision.ImageAnnotatorClient()
    with open(infile, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    text = response.full_text_annotation.text
    print(f'Detected text: {text}')
    return text

def create_glossary(languages: list, project_id: str, glossary_name: str, glossary_uri: str) -> str:
    if False:
        i = 10
        return i + 15
    "Creates a GCP glossary resource\n    Assumes you've already manually uploaded a glossary to Cloud Storage\n\n    Args:\n    languages: list of languages in the glossary\n    project_id: GCP project id\n    glossary_name: name you want to give this glossary resource\n    glossary_uri: the uri of the glossary you uploaded to Cloud Storage\n\n    Returns:\n    name of the created or existing glossary\n    "
    client = translate.TranslationServiceClient()
    location = 'us-central1'
    name = client.glossary_path(project_id, location, glossary_name)
    language_codes_set = translate.Glossary.LanguageCodesSet(language_codes=languages)
    gcs_source = translate.GcsSource(input_uri=glossary_uri)
    input_config = translate.GlossaryInputConfig(gcs_source=gcs_source)
    glossary = translate.Glossary(name=name, language_codes_set=language_codes_set, input_config=input_config)
    parent = f'projects/{project_id}/locations/{location}'
    try:
        operation = client.create_glossary(parent=parent, glossary=glossary)
        operation.result(timeout=90)
        print('Created glossary ' + glossary_name + '.')
    except AlreadyExists:
        print('The glossary ' + glossary_name + ' already exists. No new glossary was created.')
    return glossary_name

def translate_text(text: str, source_language_code: str, target_language_code: str, project_id: str, glossary_name: str) -> str:
    if False:
        print('Hello World!')
    "Translates text to a given language using a glossary\n\n    Args:\n    text: String of text to translate\n    source_language_code: language of input text\n    target_language_code: language of output text\n    project_id: GCP project id\n    glossary_name: name you gave your project's glossary\n        resource when you created it\n\n    Return:\n    String of translated text\n    "
    client = translate.TranslationServiceClient()
    location = 'us-central1'
    glossary = client.glossary_path(project_id, location, glossary_name)
    glossary_config = translate.TranslateTextGlossaryConfig(glossary=glossary)
    parent = f'projects/{project_id}/locations/{location}'
    result = client.translate_text(request={'parent': parent, 'contents': [text], 'mime_type': 'text/plain', 'source_language_code': source_language_code, 'target_language_code': target_language_code, 'glossary_config': glossary_config})
    return result.glossary_translations[0].translated_text

def text_to_speech(text: str, outfile: str) -> str:
    if False:
        print('Hello World!')
    'Converts plaintext to SSML and\n    generates synthetic audio from SSML\n\n    Args:\n\n    text: text to synthesize\n    outfile: filename to use to store synthetic audio\n\n    Returns:\n    String of synthesized audio\n    '
    escaped_lines = html.escape(text)
    ssml = '<speak>{}</speak>'.format(escaped_lines.replace('\n', '\n<break time="2s"/>'))
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
    voice = texttospeech.VoiceSelectionParams(language_code='en-US', ssml_gender=texttospeech.SsmlVoiceGender.MALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    request = texttospeech.SynthesizeSpeechRequest(input=synthesis_input, voice=voice, audio_config=audio_config)
    response = client.synthesize_speech(request=request)
    with open(outfile, 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file ' + outfile)
    return outfile

def main() -> None:
    if False:
        print('Hello World!')
    'This method is called when the tutorial is run in the Google Cloud\n    Translation API. It creates a glossary, translates text to\n    French, and speaks the translated text.\n\n    Args:\n    None\n\n    Returns:\n    None\n    '
    infile = 'resources/example.png'
    outfile = 'resources/example.mp3'
    glossary_langs = ['fr', 'en']
    glossary_name = 'bistro-glossary'
    glossary_uri = 'gs://cloud-samples-data/translation/bistro_glossary.csv'
    created_glossary_name = create_glossary(glossary_langs, PROJECT_ID, glossary_name, glossary_uri)
    text_to_translate = pic_to_text(infile)
    text_to_speak = translate_text(text_to_translate, 'fr', 'en', PROJECT_ID, created_glossary_name)
    text_to_speech(text_to_speak, outfile)
if __name__ == '__main__':
    main()