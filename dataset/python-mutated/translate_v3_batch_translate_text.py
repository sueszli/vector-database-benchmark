from google.cloud import translate

def batch_translate_text(input_uri: str='gs://YOUR_BUCKET_ID/path/to/your/file.txt', output_uri: str='gs://YOUR_BUCKET_ID/path/to/save/results/', project_id: str='YOUR_PROJECT_ID', timeout: int=180) -> translate.TranslateTextResponse:
    if False:
        for i in range(10):
            print('nop')
    'Translates a batch of texts on GCS and stores the result in a GCS location.\n\n    Args:\n        input_uri: The input URI of the texts to be translated.\n        output_uri: The output URI of the translated texts.\n        project_id: The ID of the project that owns the destination bucket.\n        timeout: The timeout for this batch translation operation.\n\n    Returns:\n        The translated texts.\n    '
    client = translate.TranslationServiceClient()
    location = 'us-central1'
    gcs_source = {'input_uri': input_uri}
    input_configs_element = {'gcs_source': gcs_source, 'mime_type': 'text/plain'}
    gcs_destination = {'output_uri_prefix': output_uri}
    output_config = {'gcs_destination': gcs_destination}
    parent = f'projects/{project_id}/locations/{location}'
    operation = client.batch_translate_text(request={'parent': parent, 'source_language_code': 'en', 'target_language_codes': ['ja'], 'input_configs': [input_configs_element], 'output_config': output_config})
    print('Waiting for operation to complete...')
    response = operation.result(timeout)
    print(f'Total Characters: {response.total_characters}')
    print(f'Translated Characters: {response.translated_characters}')
    return response