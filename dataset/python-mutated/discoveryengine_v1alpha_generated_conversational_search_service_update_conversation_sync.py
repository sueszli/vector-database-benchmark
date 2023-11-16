from google.cloud import discoveryengine_v1alpha

def sample_update_conversation():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1alpha.ConversationalSearchServiceClient()
    request = discoveryengine_v1alpha.UpdateConversationRequest()
    response = client.update_conversation(request=request)
    print(response)