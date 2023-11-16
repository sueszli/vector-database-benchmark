from google.cloud import dialogflow_v2beta1

def sample_compile_suggestion():
    if False:
        return 10
    client = dialogflow_v2beta1.ParticipantsClient()
    request = dialogflow_v2beta1.CompileSuggestionRequest()
    response = client.compile_suggestion(request=request)
    print(response)