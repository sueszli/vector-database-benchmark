from google.cloud import discoveryengine_v1alpha

def sample_list_conversations():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1alpha.ConversationalSearchServiceClient()
    request = discoveryengine_v1alpha.ListConversationsRequest(parent='parent_value')
    page_result = client.list_conversations(request=request)
    for response in page_result:
        print(response)