from google.cloud import translate

def detect_language(project_id: str='YOUR_PROJECT_ID') -> translate.DetectLanguageResponse:
    if False:
        for i in range(10):
            print('nop')
    'Detecting the language of a text string.\n\n    Args:\n        project_id: The GCP project ID.\n\n    Returns:\n        The detected language of the text.\n    '
    client = translate.TranslationServiceClient()
    location = 'global'
    parent = f'projects/{project_id}/locations/{location}'
    response = client.detect_language(content='Hello, world!', parent=parent, mime_type='text/plain')
    for language in response.languages:
        print(f'Language code: {language.language_code}')
        print(f'Confidence: {language.confidence}')
    return response