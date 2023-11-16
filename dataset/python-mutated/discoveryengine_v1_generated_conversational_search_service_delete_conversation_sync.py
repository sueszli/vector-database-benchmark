from google.cloud import discoveryengine_v1

def sample_delete_conversation():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1.ConversationalSearchServiceClient()
    request = discoveryengine_v1.DeleteConversationRequest(name='name_value')
    client.delete_conversation(request=request)