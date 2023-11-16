from google.cloud import translate_v3beta1 as translate

def batch_translate_document(input_uri: str, output_uri: str, project_id: str, timeout: int=180) -> translate.BatchTranslateDocumentResponse:
    if False:
        for i in range(10):
            print('nop')
    'Batch translate documents.\n\n    Args:\n        input_uri: Google Cloud Storage location of the input document.\n        output_uri: Google Cloud Storage location of the output document.\n        project_id: The GCP project ID.\n        timeout: The timeout for this request.\n\n    Returns:\n        Translated document response\n    '
    client = translate.TranslationServiceClient()
    location = 'us-central1'
    gcs_source = {'input_uri': input_uri}
    batch_document_input_configs = {'gcs_source': gcs_source}
    gcs_destination = {'output_uri_prefix': output_uri}
    batch_document_output_config = {'gcs_destination': gcs_destination}
    parent = f'projects/{project_id}/locations/{location}'
    operation = client.batch_translate_document(request={'parent': parent, 'source_language_code': 'en-US', 'target_language_codes': ['fr-FR'], 'input_configs': [batch_document_input_configs], 'output_config': batch_document_output_config})
    print('Waiting for operation to complete...')
    response = operation.result(timeout)
    print(f'Total Pages: {response.total_pages}')
    return response