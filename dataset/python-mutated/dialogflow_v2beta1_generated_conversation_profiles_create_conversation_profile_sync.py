from google.cloud import dialogflow_v2beta1

def sample_create_conversation_profile():
    if False:
        return 10
    client = dialogflow_v2beta1.ConversationProfilesClient()
    conversation_profile = dialogflow_v2beta1.ConversationProfile()
    conversation_profile.display_name = 'display_name_value'
    request = dialogflow_v2beta1.CreateConversationProfileRequest(parent='parent_value', conversation_profile=conversation_profile)
    response = client.create_conversation_profile(request=request)
    print(response)