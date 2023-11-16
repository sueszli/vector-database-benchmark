from google.cloud import translate_v3

def sample_delete_glossary():
    if False:
        i = 10
        return i + 15
    client = translate_v3.TranslationServiceClient()
    request = translate_v3.DeleteGlossaryRequest(name='name_value')
    operation = client.delete_glossary(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)