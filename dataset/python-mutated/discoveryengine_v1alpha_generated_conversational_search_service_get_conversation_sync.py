from google.cloud import discoveryengine_v1alpha

def sample_get_conversation():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1alpha.ConversationalSearchServiceClient()
    request = discoveryengine_v1alpha.GetConversationRequest(name='name_value')
    response = client.get_conversation(request=request)
    print(response)