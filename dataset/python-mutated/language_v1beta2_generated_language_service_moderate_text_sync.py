from google.cloud import language_v1beta2

def sample_moderate_text():
    if False:
        while True:
            i = 10
    client = language_v1beta2.LanguageServiceClient()
    document = language_v1beta2.Document()
    document.content = 'content_value'
    request = language_v1beta2.ModerateTextRequest(document=document)
    response = client.moderate_text(request=request)
    print(response)