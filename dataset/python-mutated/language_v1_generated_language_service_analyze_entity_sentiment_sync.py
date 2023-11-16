from google.cloud import language_v1

def sample_analyze_entity_sentiment():
    if False:
        i = 10
        return i + 15
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document()
    document.content = 'content_value'
    request = language_v1.AnalyzeEntitySentimentRequest(document=document)
    response = client.analyze_entity_sentiment(request=request)
    print(response)