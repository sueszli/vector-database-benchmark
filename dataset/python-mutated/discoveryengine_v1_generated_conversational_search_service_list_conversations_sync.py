from google.cloud import discoveryengine_v1

def sample_list_conversations():
    if False:
        print('Hello World!')
    client = discoveryengine_v1.ConversationalSearchServiceClient()
    request = discoveryengine_v1.ListConversationsRequest(parent='parent_value')
    page_result = client.list_conversations(request=request)
    for response in page_result:
        print(response)