from google.cloud import language_v1

def sample_analyze_entities():
    if False:
        while True:
            i = 10
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document()
    document.content = 'content_value'
    request = language_v1.AnalyzeEntitiesRequest(document=document)
    response = client.analyze_entities(request=request)
    print(response)