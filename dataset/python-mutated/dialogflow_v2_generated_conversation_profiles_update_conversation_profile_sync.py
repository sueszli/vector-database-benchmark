from google.cloud import dialogflow_v2

def sample_update_conversation_profile():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.ConversationProfilesClient()
    conversation_profile = dialogflow_v2.ConversationProfile()
    conversation_profile.display_name = 'display_name_value'
    request = dialogflow_v2.UpdateConversationProfileRequest(conversation_profile=conversation_profile)
    response = client.update_conversation_profile(request=request)
    print(response)