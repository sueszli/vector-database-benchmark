from google.cloud import discoveryengine_v1alpha

def sample_delete_conversation():
    if False:
        return 10
    client = discoveryengine_v1alpha.ConversationalSearchServiceClient()
    request = discoveryengine_v1alpha.DeleteConversationRequest(name='name_value')
    client.delete_conversation(request=request)