from google.cloud import dialogflow_v2beta1

def sample_list_suggestions():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.ParticipantsClient()
    request = dialogflow_v2beta1.ListSuggestionsRequest()
    page_result = client.list_suggestions(request=request)
    for response in page_result:
        print(response)