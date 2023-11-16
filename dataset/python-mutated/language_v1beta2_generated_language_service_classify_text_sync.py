from google.cloud import language_v1beta2

def sample_classify_text():
    if False:
        print('Hello World!')
    client = language_v1beta2.LanguageServiceClient()
    document = language_v1beta2.Document()
    document.content = 'content_value'
    request = language_v1beta2.ClassifyTextRequest(document=document)
    response = client.classify_text(request=request)
    print(response)