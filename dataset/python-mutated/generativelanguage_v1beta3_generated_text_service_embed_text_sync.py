from google.ai import generativelanguage_v1beta3

def sample_embed_text():
    if False:
        i = 10
        return i + 15
    client = generativelanguage_v1beta3.TextServiceClient()
    request = generativelanguage_v1beta3.EmbedTextRequest(model='model_value', text='text_value')
    response = client.embed_text(request=request)
    print(response)