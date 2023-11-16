from google.cloud import dialogflow_v2beta1

def sample_delete_conversation_profile():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.ConversationProfilesClient()
    request = dialogflow_v2beta1.DeleteConversationProfileRequest(name='name_value')
    client.delete_conversation_profile(request=request)