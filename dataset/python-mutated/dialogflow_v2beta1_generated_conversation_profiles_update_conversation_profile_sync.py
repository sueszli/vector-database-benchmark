from google.cloud import dialogflow_v2beta1

def sample_update_conversation_profile():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.ConversationProfilesClient()
    conversation_profile = dialogflow_v2beta1.ConversationProfile()
    conversation_profile.display_name = 'display_name_value'
    request = dialogflow_v2beta1.UpdateConversationProfileRequest(conversation_profile=conversation_profile)
    response = client.update_conversation_profile(request=request)
    print(response)