from google.cloud import discoveryengine_v1

def sample_create_conversation():
    if False:
        print('Hello World!')
    client = discoveryengine_v1.ConversationalSearchServiceClient()
    request = discoveryengine_v1.CreateConversationRequest(parent='parent_value')
    response = client.create_conversation(request=request)
    print(response)