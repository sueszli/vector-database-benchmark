from google.ai import generativelanguage_v1beta2

def sample_embed_text():
    if False:
        print('Hello World!')
    client = generativelanguage_v1beta2.TextServiceClient()
    request = generativelanguage_v1beta2.EmbedTextRequest(model='model_value', text='text_value')
    response = client.embed_text(request=request)
    print(response)