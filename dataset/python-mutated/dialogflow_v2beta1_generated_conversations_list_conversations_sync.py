from google.cloud import dialogflow_v2beta1

def sample_list_conversations():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.ConversationsClient()
    request = dialogflow_v2beta1.ListConversationsRequest(parent='parent_value')
    page_result = client.list_conversations(request=request)
    for response in page_result:
        print(response)