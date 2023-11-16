from google.cloud import translate

def batch_translate_text_with_glossary(input_uri: str='gs://YOUR_BUCKET_ID/path/to/your/file.txt', output_uri: str='gs://YOUR_BUCKET_ID/path/to/save/results/', project_id: str='YOUR_PROJECT_ID', glossary_id: str='YOUR_GLOSSARY_ID', timeout: int=320) -> translate.TranslateTextResponse:
    if False:
        for i in range(10):
            print('nop')
    'Translates a batch of texts on GCS and stores the result in a GCS location.\n    Glossary is applied for translation.\n\n    Args:\n        input_uri (str): The input file to translate.\n        output_uri (str): The output file to save the translations to.\n        project_id (str): The ID of the GCP project that owns the location.\n        glossary_id (str): The ID of the glossary to use.\n        timeout (int): The amount of time, in seconds, to wait for the operation to complete.\n\n    Returns:\n        The response from the batch.\n    '
    client = translate.TranslationServiceClient()
    location = 'us-central1'
    gcs_source = {'input_uri': input_uri}
    input_configs_element = {'gcs_source': gcs_source, 'mime_type': 'text/plain'}
    gcs_destination = {'output_uri_prefix': output_uri}
    output_config = {'gcs_destination': gcs_destination}
    parent = f'projects/{project_id}/locations/{location}'
    glossary_path = client.glossary_path(project_id, 'us-central1', glossary_id)
    glossary_config = translate.TranslateTextGlossaryConfig(glossary=glossary_path)
    glossaries = {'ja': glossary_config}
    operation = client.batch_translate_text(request={'parent': parent, 'source_language_code': 'en', 'target_language_codes': ['ja'], 'input_configs': [input_configs_element], 'glossaries': glossaries, 'output_config': output_config})
    print('Waiting for operation to complete...')
    response = operation.result(timeout)
    print(f'Total Characters: {response.total_characters}')
    print(f'Translated Characters: {response.translated_characters}')
    return response