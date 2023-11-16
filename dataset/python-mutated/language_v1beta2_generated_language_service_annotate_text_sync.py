from google.cloud import language_v1beta2

def sample_annotate_text():
    if False:
        print('Hello World!')
    client = language_v1beta2.LanguageServiceClient()
    document = language_v1beta2.Document()
    document.content = 'content_value'
    request = language_v1beta2.AnnotateTextRequest(document=document)
    response = client.annotate_text(request=request)
    print(response)