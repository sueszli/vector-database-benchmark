from google.ai import generativelanguage_v1beta3

def sample_count_text_tokens():
    if False:
        return 10
    client = generativelanguage_v1beta3.TextServiceClient()
    prompt = generativelanguage_v1beta3.TextPrompt()
    prompt.text = 'text_value'
    request = generativelanguage_v1beta3.CountTextTokensRequest(model='model_value', prompt=prompt)
    response = client.count_text_tokens(request=request)
    print(response)