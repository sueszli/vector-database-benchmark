from google.cloud import dialogflow_v2

def sample_suggest_articles():
    if False:
        return 10
    client = dialogflow_v2.ParticipantsClient()
    request = dialogflow_v2.SuggestArticlesRequest(parent='parent_value')
    response = client.suggest_articles(request=request)
    print(response)