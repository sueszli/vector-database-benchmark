from google.cloud import dialogflowcx_v3

def sample_streaming_detect_intent():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.SessionsClient()
    query_input = dialogflowcx_v3.QueryInput()
    query_input.text.text = 'text_value'
    query_input.language_code = 'language_code_value'
    request = dialogflowcx_v3.StreamingDetectIntentRequest(query_input=query_input)
    requests = [request]

    def request_generator():
        if False:
            print('Hello World!')
        for request in requests:
            yield request
    stream = client.streaming_detect_intent(requests=request_generator())
    for response in stream:
        print(response)