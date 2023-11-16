from google.cloud import language_v2

def sample_annotate_text():
    if False:
        return 10
    client = language_v2.LanguageServiceClient()
    document = language_v2.Document()
    document.content = 'content_value'
    request = language_v2.AnnotateTextRequest(document=document)
    response = client.annotate_text(request=request)
    print(response)