from google.cloud import discoveryengine_v1alpha

def sample_create_conversation():
    if False:
        print('Hello World!')
    client = discoveryengine_v1alpha.ConversationalSearchServiceClient()
    request = discoveryengine_v1alpha.CreateConversationRequest(parent='parent_value')
    response = client.create_conversation(request=request)
    print(response)