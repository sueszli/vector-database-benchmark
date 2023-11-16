from google.cloud import translate_v3beta1

def sample_delete_glossary():
    if False:
        for i in range(10):
            print('nop')
    client = translate_v3beta1.TranslationServiceClient()
    request = translate_v3beta1.DeleteGlossaryRequest(name='name_value')
    operation = client.delete_glossary(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)