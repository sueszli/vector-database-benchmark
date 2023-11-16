from google.cloud import translate_v3beta1

def sample_translate_text():
    if False:
        i = 10
        return i + 15
    client = translate_v3beta1.TranslationServiceClient()
    request = translate_v3beta1.TranslateTextRequest(contents=['contents_value1', 'contents_value2'], target_language_code='target_language_code_value', parent='parent_value')
    response = client.translate_text(request=request)
    print(response)