from google.cloud import discoveryengine_v1

def sample_update_conversation():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1.ConversationalSearchServiceClient()
    request = discoveryengine_v1.UpdateConversationRequest()
    response = client.update_conversation(request=request)
    print(response)