from google.ai import generativelanguage_v1beta3

def sample_generate_message():
    if False:
        while True:
            i = 10
    client = generativelanguage_v1beta3.DiscussServiceClient()
    prompt = generativelanguage_v1beta3.MessagePrompt()
    prompt.messages.content = 'content_value'
    request = generativelanguage_v1beta3.GenerateMessageRequest(model='model_value', prompt=prompt)
    response = client.generate_message(request=request)
    print(response)