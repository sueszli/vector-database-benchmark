from google.cloud import translate_v3beta1

def sample_detect_language():
    if False:
        print('Hello World!')
    client = translate_v3beta1.TranslationServiceClient()
    request = translate_v3beta1.DetectLanguageRequest(content='content_value', parent='parent_value')
    response = client.detect_language(request=request)
    print(response)