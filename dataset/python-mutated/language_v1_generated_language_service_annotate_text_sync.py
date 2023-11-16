from google.cloud import language_v1

def sample_annotate_text():
    if False:
        i = 10
        return i + 15
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document()
    document.content = 'content_value'
    request = language_v1.AnnotateTextRequest(document=document)
    response = client.annotate_text(request=request)
    print(response)