from google.cloud import translate

def batch_translate_text_with_glossary_and_model(input_uri: str, output_uri: str, project_id: str, model_id: str, glossary_id: str) -> translate.TranslateTextResponse:
    if False:
        print('Hello World!')
    'Batch translate text with Glossary and Translation model.\n    Args:\n        input_uri: The input text to be translated.\n        output_uri: The output text to be translated.\n        project_id: The ID of the GCP project that owns the model.\n        model_id: The ID of the model\n        glossary_id: The ID of the glossary\n\n    Returns:\n        The translated text.\n    '
    client = translate.TranslationServiceClient()
    location = 'us-central1'
    target_language_codes = ['ja']
    gcs_source = {'input_uri': input_uri}
    mime_type = 'text/plain'
    input_configs_element = {'gcs_source': gcs_source, 'mime_type': mime_type}
    input_configs = [input_configs_element]
    gcs_destination = {'output_uri_prefix': output_uri}
    output_config = {'gcs_destination': gcs_destination}
    parent = f'projects/{project_id}/locations/{location}'
    model_path = 'projects/{}/locations/{}/models/{}'.format(project_id, 'us-central1', model_id)
    models = {'ja': model_path}
    glossary_path = client.glossary_path(project_id, 'us-central1', glossary_id)
    glossary_config = translate.TranslateTextGlossaryConfig(glossary=glossary_path)
    glossaries = {'ja': glossary_config}
    operation = client.batch_translate_text(request={'parent': parent, 'source_language_code': 'en', 'target_language_codes': target_language_codes, 'input_configs': input_configs, 'output_config': output_config, 'models': models, 'glossaries': glossaries})
    print('Waiting for operation to complete...')
    response = operation.result()
    print(f'Total Characters: {response.total_characters}')
    print(f'Translated Characters: {response.translated_characters}')
    return response