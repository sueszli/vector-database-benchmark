from google.cloud import language_v2

def sample_classify_text():
    if False:
        for i in range(10):
            print('nop')
    client = language_v2.LanguageServiceClient()
    document = language_v2.Document()
    document.content = 'content_value'
    request = language_v2.ClassifyTextRequest(document=document)
    response = client.classify_text(request=request)
    print(response)