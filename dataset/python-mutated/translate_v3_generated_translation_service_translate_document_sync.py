from google.cloud import translate_v3

def sample_translate_document():
    if False:
        while True:
            i = 10
    client = translate_v3.TranslationServiceClient()
    document_input_config = translate_v3.DocumentInputConfig()
    document_input_config.content = b'content_blob'
    request = translate_v3.TranslateDocumentRequest(parent='parent_value', target_language_code='target_language_code_value', document_input_config=document_input_config)
    response = client.translate_document(request=request)
    print(response)