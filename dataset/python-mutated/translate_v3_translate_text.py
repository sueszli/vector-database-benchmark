from google.cloud import translate

def translate_text(text: str='YOUR_TEXT_TO_TRANSLATE', project_id: str='YOUR_PROJECT_ID') -> translate.TranslationServiceClient:
    if False:
        print('Hello World!')
    'Translating Text.'
    client = translate.TranslationServiceClient()
    location = 'global'
    parent = f'projects/{project_id}/locations/{location}'
    response = client.translate_text(request={'parent': parent, 'contents': [text], 'mime_type': 'text/plain', 'source_language_code': 'en-US', 'target_language_code': 'fr'})
    for translation in response.translations:
        print(f'Translated text: {translation.translated_text}')
    return response