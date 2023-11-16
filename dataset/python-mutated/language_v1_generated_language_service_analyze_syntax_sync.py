from google.cloud import language_v1

def sample_analyze_syntax():
    if False:
        return 10
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document()
    document.content = 'content_value'
    request = language_v1.AnalyzeSyntaxRequest(document=document)
    response = client.analyze_syntax(request=request)
    print(response)