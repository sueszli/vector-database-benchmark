from google.cloud import dialogflow_v2beta1

def sample_set_suggestion_feature_config():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.ConversationProfilesClient()
    request = dialogflow_v2beta1.SetSuggestionFeatureConfigRequest(conversation_profile='conversation_profile_value', participant_role='END_USER')
    operation = client.set_suggestion_feature_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)