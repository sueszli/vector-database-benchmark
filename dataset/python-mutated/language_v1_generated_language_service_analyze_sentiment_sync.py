from google.cloud import language_v1

def sample_analyze_sentiment():
    if False:
        for i in range(10):
            print('nop')
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document()
    document.content = 'content_value'
    request = language_v1.AnalyzeSentimentRequest(document=document)
    response = client.analyze_sentiment(request=request)
    print(response)