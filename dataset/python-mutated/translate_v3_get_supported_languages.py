from google.cloud import translate

def get_supported_languages(project_id: str='YOUR_PROJECT_ID') -> translate.SupportedLanguages:
    if False:
        return 10
    'Getting a list of supported language codes.\n\n    Args:\n        project_id: The GCP project ID.\n\n    Returns:\n        A list of supported language codes.\n    '
    client = translate.TranslationServiceClient()
    parent = f'projects/{project_id}'
    response = client.get_supported_languages(parent=parent)
    print('Supported Languages:')
    for language in response.languages:
        print(f'Language Code: {language.language_code}')
    return response