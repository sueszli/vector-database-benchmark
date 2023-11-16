from google.cloud import translate_v3beta1

def sample_translate_document():
    if False:
        i = 10
        return i + 15
    client = translate_v3beta1.TranslationServiceClient()
    document_input_config = translate_v3beta1.DocumentInputConfig()
    document_input_config.content = b'content_blob'
    request = translate_v3beta1.TranslateDocumentRequest(parent='parent_value', target_language_code='target_language_code_value', document_input_config=document_input_config)
    response = client.translate_document(request=request)
    print(response)