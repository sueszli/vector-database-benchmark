from google.cloud import translate

def list_glossaries(project_id: str='YOUR_PROJECT_ID') -> translate.Glossary:
    if False:
        i = 10
        return i + 15
    'List Glossaries.\n\n    Args:\n        project_id: The GCP project ID.\n\n    Returns:\n        The glossary.\n    '
    client = translate.TranslationServiceClient()
    location = 'us-central1'
    parent = f'projects/{project_id}/locations/{location}'
    for glossary in client.list_glossaries(parent=parent):
        print(f'Name: {glossary.name}')
        print(f'Entry count: {glossary.entry_count}')
        print(f'Input uri: {glossary.input_config.gcs_source.input_uri}')
        for language_code in glossary.language_codes_set.language_codes:
            print(f'Language code: {language_code}')
    return glossary