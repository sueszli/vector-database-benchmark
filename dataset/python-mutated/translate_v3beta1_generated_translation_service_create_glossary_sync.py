from google.cloud import translate_v3beta1

def sample_create_glossary():
    if False:
        for i in range(10):
            print('nop')
    client = translate_v3beta1.TranslationServiceClient()
    glossary = translate_v3beta1.Glossary()
    glossary.name = 'name_value'
    request = translate_v3beta1.CreateGlossaryRequest(parent='parent_value', glossary=glossary)
    operation = client.create_glossary(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)