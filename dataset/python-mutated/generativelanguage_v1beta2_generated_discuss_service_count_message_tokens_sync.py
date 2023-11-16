from google.ai import generativelanguage_v1beta2

def sample_count_message_tokens():
    if False:
        while True:
            i = 10
    client = generativelanguage_v1beta2.DiscussServiceClient()
    prompt = generativelanguage_v1beta2.MessagePrompt()
    prompt.messages.content = 'content_value'
    request = generativelanguage_v1beta2.CountMessageTokensRequest(model='model_value', prompt=prompt)
    response = client.count_message_tokens(request=request)
    print(response)