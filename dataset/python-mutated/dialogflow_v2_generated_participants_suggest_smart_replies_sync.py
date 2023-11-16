from google.cloud import dialogflow_v2

def sample_suggest_smart_replies():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.ParticipantsClient()
    request = dialogflow_v2.SuggestSmartRepliesRequest(parent='parent_value')
    response = client.suggest_smart_replies(request=request)
    print(response)