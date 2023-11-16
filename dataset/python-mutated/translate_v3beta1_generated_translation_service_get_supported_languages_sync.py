from google.cloud import translate_v3beta1

def sample_get_supported_languages():
    if False:
        for i in range(10):
            print('nop')
    client = translate_v3beta1.TranslationServiceClient()
    request = translate_v3beta1.GetSupportedLanguagesRequest(parent='parent_value')
    response = client.get_supported_languages(request=request)
    print(response)