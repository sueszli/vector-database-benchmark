from google.cloud import dialogflow_v2beta1

def sample_batch_create_messages():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.ConversationsClient()
    requests = dialogflow_v2beta1.CreateMessageRequest()
    requests.parent = 'parent_value'
    requests.message.content = 'content_value'
    request = dialogflow_v2beta1.BatchCreateMessagesRequest(parent='parent_value', requests=requests)
    response = client.batch_create_messages(request=request)
    print(response)