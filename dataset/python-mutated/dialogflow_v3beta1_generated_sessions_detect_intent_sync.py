from google.cloud import dialogflowcx_v3beta1

def sample_detect_intent():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.SessionsClient()
    query_input = dialogflowcx_v3beta1.QueryInput()
    query_input.text.text = 'text_value'
    query_input.language_code = 'language_code_value'
    request = dialogflowcx_v3beta1.DetectIntentRequest(session='session_value', query_input=query_input)
    response = client.detect_intent(request=request)
    print(response)