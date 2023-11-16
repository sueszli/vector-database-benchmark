from google.cloud import translate_v3

def sample_get_glossary():
    if False:
        for i in range(10):
            print('nop')
    client = translate_v3.TranslationServiceClient()
    request = translate_v3.GetGlossaryRequest(name='name_value')
    response = client.get_glossary(request=request)
    print(response)