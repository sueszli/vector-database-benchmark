from google.cloud import dialogflow_v2beta1

def sample_suggest_articles():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.ParticipantsClient()
    request = dialogflow_v2beta1.SuggestArticlesRequest(parent='parent_value')
    response = client.suggest_articles(request=request)
    print(response)