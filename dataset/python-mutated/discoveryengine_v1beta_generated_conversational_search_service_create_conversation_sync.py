from google.cloud import discoveryengine_v1beta

def sample_create_conversation():
    if False:
        print('Hello World!')
    client = discoveryengine_v1beta.ConversationalSearchServiceClient()
    request = discoveryengine_v1beta.CreateConversationRequest(parent='parent_value')
    response = client.create_conversation(request=request)
    print(response)