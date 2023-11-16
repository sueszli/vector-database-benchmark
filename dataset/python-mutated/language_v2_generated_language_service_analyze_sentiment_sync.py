from google.cloud import language_v2

def sample_analyze_sentiment():
    if False:
        print('Hello World!')
    client = language_v2.LanguageServiceClient()
    document = language_v2.Document()
    document.content = 'content_value'
    request = language_v2.AnalyzeSentimentRequest(document=document)
    response = client.analyze_sentiment(request=request)
    print(response)