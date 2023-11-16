from google.cloud import translate_v3

def sample_detect_language():
    if False:
        while True:
            i = 10
    client = translate_v3.TranslationServiceClient()
    request = translate_v3.DetectLanguageRequest(content='content_value', parent='parent_value')
    response = client.detect_language(request=request)
    print(response)