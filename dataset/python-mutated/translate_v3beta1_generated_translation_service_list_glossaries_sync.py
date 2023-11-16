from google.cloud import translate_v3beta1

def sample_list_glossaries():
    if False:
        return 10
    client = translate_v3beta1.TranslationServiceClient()
    request = translate_v3beta1.ListGlossariesRequest(parent='parent_value')
    page_result = client.list_glossaries(request=request)
    for response in page_result:
        print(response)