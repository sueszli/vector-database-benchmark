from google.cloud import translate

def translate_text_with_model(text: str='YOUR_TEXT_TO_TRANSLATE', project_id: str='YOUR_PROJECT_ID', model_id: str='YOUR_MODEL_ID') -> translate.TranslationServiceClient:
    if False:
        while True:
            i = 10
    'Translates a given text using Translation custom model.'
    client = translate.TranslationServiceClient()
    location = 'us-central1'
    parent = f'projects/{project_id}/locations/{location}'
    model_path = f'{parent}/models/{model_id}'
    response = client.translate_text(request={'contents': [text], 'target_language_code': 'ja', 'model': model_path, 'source_language_code': 'en', 'parent': parent, 'mime_type': 'text/plain'})
    for translation in response.translations:
        print(f'Translated text: {translation.translated_text}')
    return response