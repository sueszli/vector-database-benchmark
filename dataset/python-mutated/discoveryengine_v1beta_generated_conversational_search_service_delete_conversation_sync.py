from google.cloud import discoveryengine_v1beta

def sample_delete_conversation():
    if False:
        print('Hello World!')
    client = discoveryengine_v1beta.ConversationalSearchServiceClient()
    request = discoveryengine_v1beta.DeleteConversationRequest(name='name_value')
    client.delete_conversation(request=request)