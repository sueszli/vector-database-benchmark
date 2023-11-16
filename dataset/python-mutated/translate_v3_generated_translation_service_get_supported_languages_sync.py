from google.cloud import translate_v3

def sample_get_supported_languages():
    if False:
        while True:
            i = 10
    client = translate_v3.TranslationServiceClient()
    request = translate_v3.GetSupportedLanguagesRequest(parent='parent_value')
    response = client.get_supported_languages(request=request)
    print(response)