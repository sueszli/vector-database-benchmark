from google.cloud import translate

def batch_translate_text_with_model(input_uri: str='gs://YOUR_BUCKET_ID/path/to/your/file.txt', output_uri: str='gs://YOUR_BUCKET_ID/path/to/save/results/', project_id: str='YOUR_PROJECT_ID', model_id: str='YOUR_MODEL_ID') -> translate.TranslationServiceClient:
    if False:
        while True:
            i = 10
    'Batch translate text using Translation model.\n    Model can be AutoML or General[built-in] model.\n\n    Args:\n        input_uri: The input file to translate.\n        output_uri: The output file to save the translation results.\n        project_id: The ID of the GCP project that owns the model.\n        model_id: The model ID.\n\n    Returns:\n        The response from the batch translation API.\n    '
    client = translate.TranslationServiceClient()
    gcs_source = {'input_uri': input_uri}
    location = 'us-central1'
    input_configs_element = {'gcs_source': gcs_source, 'mime_type': 'text/plain'}
    gcs_destination = {'output_uri_prefix': output_uri}
    output_config = {'gcs_destination': gcs_destination}
    parent = f'projects/{project_id}/locations/{location}'
    model_path = 'projects/{}/locations/{}/models/{}'.format(project_id, location, model_id)
    models = {'ja': model_path}
    operation = client.batch_translate_text(request={'parent': parent, 'source_language_code': 'en', 'target_language_codes': ['ja'], 'input_configs': [input_configs_element], 'output_config': output_config, 'models': models})
    print('Waiting for operation to complete...')
    response = operation.result()
    print(f'Total Characters: {response.total_characters}')
    print(f'Translated Characters: {response.translated_characters}')
    return response