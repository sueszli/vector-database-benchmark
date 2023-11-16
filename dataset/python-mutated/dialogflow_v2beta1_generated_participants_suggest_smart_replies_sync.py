from google.cloud import dialogflow_v2beta1

def sample_suggest_smart_replies():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.ParticipantsClient()
    request = dialogflow_v2beta1.SuggestSmartRepliesRequest(parent='parent_value')
    response = client.suggest_smart_replies(request=request)
    print(response)