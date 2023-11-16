from google.cloud import language_v1beta2

def sample_analyze_entity_sentiment():
    if False:
        for i in range(10):
            print('nop')
    client = language_v1beta2.LanguageServiceClient()
    document = language_v1beta2.Document()
    document.content = 'content_value'
    request = language_v1beta2.AnalyzeEntitySentimentRequest(document=document)
    response = client.analyze_entity_sentiment(request=request)
    print(response)