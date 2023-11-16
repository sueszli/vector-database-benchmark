from google.cloud import language_v1beta2

def sample_analyze_entities():
    if False:
        print('Hello World!')
    client = language_v1beta2.LanguageServiceClient()
    document = language_v1beta2.Document()
    document.content = 'content_value'
    request = language_v1beta2.AnalyzeEntitiesRequest(document=document)
    response = client.analyze_entities(request=request)
    print(response)