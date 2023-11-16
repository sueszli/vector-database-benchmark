from google.cloud import dialogflow_v2

def sample_list_messages():
    if False:
        return 10
    client = dialogflow_v2.ConversationsClient()
    request = dialogflow_v2.ListMessagesRequest(parent='parent_value')
    page_result = client.list_messages(request=request)
    for response in page_result:
        print(response)