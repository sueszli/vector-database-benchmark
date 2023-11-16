from google.cloud import translate_v3

def sample_list_glossaries():
    if False:
        for i in range(10):
            print('nop')
    client = translate_v3.TranslationServiceClient()
    request = translate_v3.ListGlossariesRequest(parent='parent_value')
    page_result = client.list_glossaries(request=request)
    for response in page_result:
        print(response)