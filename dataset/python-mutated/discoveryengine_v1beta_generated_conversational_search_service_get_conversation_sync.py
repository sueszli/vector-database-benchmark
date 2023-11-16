from google.cloud import discoveryengine_v1beta

def sample_get_conversation():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1beta.ConversationalSearchServiceClient()
    request = discoveryengine_v1beta.GetConversationRequest(name='name_value')
    response = client.get_conversation(request=request)
    print(response)