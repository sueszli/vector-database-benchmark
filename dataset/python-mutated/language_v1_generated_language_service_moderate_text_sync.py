from google.cloud import language_v1

def sample_moderate_text():
    if False:
        return 10
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document()
    document.content = 'content_value'
    request = language_v1.ModerateTextRequest(document=document)
    response = client.moderate_text(request=request)
    print(response)