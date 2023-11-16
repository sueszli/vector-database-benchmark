from google.ai import generativelanguage_v1beta2

def sample_generate_text():
    if False:
        while True:
            i = 10
    client = generativelanguage_v1beta2.TextServiceClient()
    prompt = generativelanguage_v1beta2.TextPrompt()
    prompt.text = 'text_value'
    request = generativelanguage_v1beta2.GenerateTextRequest(model='model_value', prompt=prompt)
    response = client.generate_text(request=request)
    print(response)