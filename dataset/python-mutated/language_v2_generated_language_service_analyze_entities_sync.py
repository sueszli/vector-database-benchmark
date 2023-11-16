from google.cloud import language_v2

def sample_analyze_entities():
    if False:
        for i in range(10):
            print('nop')
    client = language_v2.LanguageServiceClient()
    document = language_v2.Document()
    document.content = 'content_value'
    request = language_v2.AnalyzeEntitiesRequest(document=document)
    response = client.analyze_entities(request=request)
    print(response)