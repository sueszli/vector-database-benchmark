from google.cloud import dialogflow_v2

def sample_analyze_content():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.ParticipantsClient()
    text_input = dialogflow_v2.TextInput()
    text_input.text = 'text_value'
    text_input.language_code = 'language_code_value'
    request = dialogflow_v2.AnalyzeContentRequest(text_input=text_input, participant='participant_value')
    response = client.analyze_content(request=request)
    print(response)