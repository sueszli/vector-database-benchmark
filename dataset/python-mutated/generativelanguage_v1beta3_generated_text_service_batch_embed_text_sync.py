from google.ai import generativelanguage_v1beta3

def sample_batch_embed_text():
    if False:
        for i in range(10):
            print('nop')
    client = generativelanguage_v1beta3.TextServiceClient()
    request = generativelanguage_v1beta3.BatchEmbedTextRequest(model='model_value', texts=['texts_value1', 'texts_value2'])
    response = client.batch_embed_text(request=request)
    print(response)