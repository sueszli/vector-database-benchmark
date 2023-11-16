from google.ai import generativelanguage_v1beta3

def sample_generate_text():
    if False:
        i = 10
        return i + 15
    client = generativelanguage_v1beta3.TextServiceClient()
    prompt = generativelanguage_v1beta3.TextPrompt()
    prompt.text = 'text_value'
    request = generativelanguage_v1beta3.GenerateTextRequest(model='model_value', prompt=prompt)
    response = client.generate_text(request=request)
    print(response)