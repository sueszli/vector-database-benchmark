from google.cloud import discoveryengine_v1beta

def sample_converse_conversation():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1beta.ConversationalSearchServiceClient()
    request = discoveryengine_v1beta.ConverseConversationRequest(name='name_value')
    response = client.converse_conversation(request=request)
    print(response)