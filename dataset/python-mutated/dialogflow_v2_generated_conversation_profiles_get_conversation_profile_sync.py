from google.cloud import dialogflow_v2

def sample_get_conversation_profile():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.ConversationProfilesClient()
    request = dialogflow_v2.GetConversationProfileRequest(name='name_value')
    response = client.get_conversation_profile(request=request)
    print(response)