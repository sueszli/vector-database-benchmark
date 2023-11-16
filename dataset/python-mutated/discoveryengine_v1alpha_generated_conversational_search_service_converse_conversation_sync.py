from google.cloud import discoveryengine_v1alpha

def sample_converse_conversation():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1alpha.ConversationalSearchServiceClient()
    request = discoveryengine_v1alpha.ConverseConversationRequest(name='name_value')
    response = client.converse_conversation(request=request)
    print(response)