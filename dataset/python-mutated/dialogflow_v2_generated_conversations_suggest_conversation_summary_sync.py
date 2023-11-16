from google.cloud import dialogflow_v2

def sample_suggest_conversation_summary():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.ConversationsClient()
    request = dialogflow_v2.SuggestConversationSummaryRequest(conversation='conversation_value')
    response = client.suggest_conversation_summary(request=request)
    print(response)