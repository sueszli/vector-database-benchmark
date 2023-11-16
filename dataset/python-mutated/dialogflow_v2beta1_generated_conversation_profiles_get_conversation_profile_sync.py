from google.cloud import dialogflow_v2beta1

def sample_get_conversation_profile():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.ConversationProfilesClient()
    request = dialogflow_v2beta1.GetConversationProfileRequest(name='name_value')
    response = client.get_conversation_profile(request=request)
    print(response)