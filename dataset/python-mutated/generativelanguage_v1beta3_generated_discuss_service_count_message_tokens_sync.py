from google.ai import generativelanguage_v1beta3

def sample_count_message_tokens():
    if False:
        i = 10
        return i + 15
    client = generativelanguage_v1beta3.DiscussServiceClient()
    prompt = generativelanguage_v1beta3.MessagePrompt()
    prompt.messages.content = 'content_value'
    request = generativelanguage_v1beta3.CountMessageTokensRequest(model='model_value', prompt=prompt)
    response = client.count_message_tokens(request=request)
    print(response)