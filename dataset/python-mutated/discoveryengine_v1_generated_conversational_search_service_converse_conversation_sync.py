from google.cloud import discoveryengine_v1

def sample_converse_conversation():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1.ConversationalSearchServiceClient()
    request = discoveryengine_v1.ConverseConversationRequest(name='name_value')
    response = client.converse_conversation(request=request)
    print(response)