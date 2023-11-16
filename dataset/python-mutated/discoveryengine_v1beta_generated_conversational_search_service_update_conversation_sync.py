from google.cloud import discoveryengine_v1beta

def sample_update_conversation():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1beta.ConversationalSearchServiceClient()
    request = discoveryengine_v1beta.UpdateConversationRequest()
    response = client.update_conversation(request=request)
    print(response)