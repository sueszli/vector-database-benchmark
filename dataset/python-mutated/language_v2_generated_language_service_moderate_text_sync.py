from google.cloud import language_v2

def sample_moderate_text():
    if False:
        print('Hello World!')
    client = language_v2.LanguageServiceClient()
    document = language_v2.Document()
    document.content = 'content_value'
    request = language_v2.ModerateTextRequest(document=document)
    response = client.moderate_text(request=request)
    print(response)