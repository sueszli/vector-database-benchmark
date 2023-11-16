from google.cloud import dialogflow_v2

def sample_delete_conversation_profile():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.ConversationProfilesClient()
    request = dialogflow_v2.DeleteConversationProfileRequest(name='name_value')
    client.delete_conversation_profile(request=request)