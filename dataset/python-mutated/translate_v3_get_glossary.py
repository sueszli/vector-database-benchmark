from google.cloud import translate_v3 as translate

def get_glossary(project_id: str='YOUR_PROJECT_ID', glossary_id: str='YOUR_GLOSSARY_ID') -> translate.Glossary:
    if False:
        while True:
            i = 10
    'Get a particular glossary based on the glossary ID.\n\n    Args:\n        project_id: The GCP project ID.\n        glossary_id: The ID of the glossary to retrieve.\n\n    Returns:\n        The glossary.\n    '
    client = translate.TranslationServiceClient()
    name = client.glossary_path(project_id, 'us-central1', glossary_id)
    response = client.get_glossary(name=name)
    print(f'Glossary name: {response.name}')
    print(f'Entry count: {response.entry_count}')
    print(f'Input URI: {response.input_config.gcs_source.input_uri}')
    return response