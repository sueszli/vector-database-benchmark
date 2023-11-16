from google.cloud import dialogflow_v2

def sample_complete_conversation():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.ConversationsClient()
    request = dialogflow_v2.CompleteConversationRequest(name='name_value')
    response = client.complete_conversation(request=request)
    print(response)