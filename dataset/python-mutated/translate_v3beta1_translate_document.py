from google.cloud import translate_v3beta1 as translate

def translate_document(project_id: str, file_path: str) -> translate.TranslationServiceClient:
    if False:
        while True:
            i = 10
    'Translates a document.\n\n    Args:\n        project_id: The GCP project ID.\n        file_path: The path to the file to be translated.\n\n    Returns:\n        The translated document.\n    '
    client = translate.TranslationServiceClient()
    location = 'us-central1'
    parent = f'projects/{project_id}/locations/{location}'
    with open(file_path, 'rb') as document:
        document_content = document.read()
    document_input_config = {'content': document_content, 'mime_type': 'application/pdf'}
    response = client.translate_document(request={'parent': parent, 'target_language_code': 'fr-FR', 'document_input_config': document_input_config})
    print(f'Response: Detected Language Code - {response.document_translation.detected_language_code}')
    return response