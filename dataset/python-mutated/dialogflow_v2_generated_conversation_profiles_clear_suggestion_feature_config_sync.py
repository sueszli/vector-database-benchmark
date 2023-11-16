from google.cloud import dialogflow_v2

def sample_clear_suggestion_feature_config():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.ConversationProfilesClient()
    request = dialogflow_v2.ClearSuggestionFeatureConfigRequest(conversation_profile='conversation_profile_value', participant_role='END_USER', suggestion_feature_type='KNOWLEDGE_SEARCH')
    operation = client.clear_suggestion_feature_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)