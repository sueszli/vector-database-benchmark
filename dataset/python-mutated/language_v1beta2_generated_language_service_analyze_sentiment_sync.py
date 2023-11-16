from google.cloud import language_v1beta2

def sample_analyze_sentiment():
    if False:
        i = 10
        return i + 15
    client = language_v1beta2.LanguageServiceClient()
    document = language_v1beta2.Document()
    document.content = 'content_value'
    request = language_v1beta2.AnalyzeSentimentRequest(document=document)
    response = client.analyze_sentiment(request=request)
    print(response)