from google.cloud import translate

def translate_text_with_glossary(text: str='YOUR_TEXT_TO_TRANSLATE', project_id: str='YOUR_PROJECT_ID', glossary_id: str='YOUR_GLOSSARY_ID') -> translate.TranslateTextResponse:
    if False:
        while True:
            i = 10
    'Translates a given text using a glossary.\n\n    Args:\n        text: The text to translate.\n        project_id: The ID of the GCP project that owns the glossary.\n        glossary_id: The ID of the glossary to use.\n\n    Returns:\n        The translated text.'
    client = translate.TranslationServiceClient()
    location = 'us-central1'
    parent = f'projects/{project_id}/locations/{location}'
    glossary = client.glossary_path(project_id, 'us-central1', glossary_id)
    glossary_config = translate.TranslateTextGlossaryConfig(glossary=glossary)
    response = client.translate_text(request={'contents': [text], 'target_language_code': 'ja', 'source_language_code': 'en', 'parent': parent, 'glossary_config': glossary_config})
    print('Translated text: \n')
    for translation in response.glossary_translations:
        print(f'\t {translation.translated_text}')
    return response