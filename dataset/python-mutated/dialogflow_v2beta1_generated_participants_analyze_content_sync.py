from google.cloud import dialogflow_v2beta1

def sample_analyze_content():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.ParticipantsClient()
    request = dialogflow_v2beta1.AnalyzeContentRequest(participant='participant_value')
    response = client.analyze_content(request=request)
    print(response)