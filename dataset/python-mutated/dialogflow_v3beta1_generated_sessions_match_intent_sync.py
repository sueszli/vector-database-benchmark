from google.cloud import dialogflowcx_v3beta1

def sample_match_intent():
    if False:
        return 10
    client = dialogflowcx_v3beta1.SessionsClient()
    query_input = dialogflowcx_v3beta1.QueryInput()
    query_input.text.text = 'text_value'
    query_input.language_code = 'language_code_value'
    request = dialogflowcx_v3beta1.MatchIntentRequest(session='session_value', query_input=query_input)
    response = client.match_intent(request=request)
    print(response)