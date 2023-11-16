from google.cloud import dialogflow_v2beta1

def sample_suggest_conversation_summary():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.ConversationsClient()
    request = dialogflow_v2beta1.SuggestConversationSummaryRequest(conversation='conversation_value')
    response = client.suggest_conversation_summary(request=request)
    print(response)