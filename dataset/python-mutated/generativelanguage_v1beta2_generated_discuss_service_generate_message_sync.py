from google.ai import generativelanguage_v1beta2

def sample_generate_message():
    if False:
        print('Hello World!')
    client = generativelanguage_v1beta2.DiscussServiceClient()
    prompt = generativelanguage_v1beta2.MessagePrompt()
    prompt.messages.content = 'content_value'
    request = generativelanguage_v1beta2.GenerateMessageRequest(model='model_value', prompt=prompt)
    response = client.generate_message(request=request)
    print(response)