from google.cloud import dialogflow_v2beta1

def sample_suggest_faq_answers():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.ParticipantsClient()
    request = dialogflow_v2beta1.SuggestFaqAnswersRequest(parent='parent_value')
    response = client.suggest_faq_answers(request=request)
    print(response)