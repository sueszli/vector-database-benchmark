from google.cloud import translate_v3

def sample_create_glossary():
    if False:
        i = 10
        return i + 15
    client = translate_v3.TranslationServiceClient()
    glossary = translate_v3.Glossary()
    glossary.name = 'name_value'
    request = translate_v3.CreateGlossaryRequest(parent='parent_value', glossary=glossary)
    operation = client.create_glossary(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)