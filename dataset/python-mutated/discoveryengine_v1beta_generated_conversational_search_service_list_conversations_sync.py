from google.cloud import discoveryengine_v1beta

def sample_list_conversations():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1beta.ConversationalSearchServiceClient()
    request = discoveryengine_v1beta.ListConversationsRequest(parent='parent_value')
    page_result = client.list_conversations(request=request)
    for response in page_result:
        print(response)