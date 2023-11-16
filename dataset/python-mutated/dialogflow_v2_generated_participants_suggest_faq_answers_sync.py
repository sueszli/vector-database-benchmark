from google.cloud import dialogflow_v2

def sample_suggest_faq_answers():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.ParticipantsClient()
    request = dialogflow_v2.SuggestFaqAnswersRequest(parent='parent_value')
    response = client.suggest_faq_answers(request=request)
    print(response)