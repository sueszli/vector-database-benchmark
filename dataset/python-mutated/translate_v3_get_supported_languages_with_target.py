from google.cloud import translate

def get_supported_languages_with_target(project_id: str='YOUR_PROJECT_ID') -> translate.SupportedLanguages:
    if False:
        while True:
            i = 10
    'Listing supported languages with target language name.\n\n    Args:\n        project_id: Your Google Cloud project ID.\n\n    Returns:\n        Supported languages.\n    '
    client = translate.TranslationServiceClient()
    location = 'global'
    parent = f'projects/{project_id}/locations/{location}'
    response = client.get_supported_languages(display_language_code='is', parent=parent)
    for language in response.languages:
        print(f'Language Code: {language.language_code}')
        print(f'Display Name: {language.display_name}')
    return response