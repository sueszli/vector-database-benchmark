from google.cloud import translate_v3beta1

def sample_get_glossary():
    if False:
        return 10
    client = translate_v3beta1.TranslationServiceClient()
    request = translate_v3beta1.GetGlossaryRequest(name='name_value')
    response = client.get_glossary(request=request)
    print(response)