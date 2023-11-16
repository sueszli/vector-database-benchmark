from google.cloud import dialogflowcx_v3beta1

def sample_streaming_detect_intent():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.SessionsClient()
    query_input = dialogflowcx_v3beta1.QueryInput()
    query_input.text.text = 'text_value'
    query_input.language_code = 'language_code_value'
    request = dialogflowcx_v3beta1.StreamingDetectIntentRequest(query_input=query_input)
    requests = [request]

    def request_generator():
        if False:
            return 10
        for request in requests:
            yield request
    stream = client.streaming_detect_intent(requests=request_generator())
    for response in stream:
        print(response)