from google.cloud import dialogflowcx_v3

async def sample_streaming_detect_intent():
    client = dialogflowcx_v3.SessionsAsyncClient()
    query_input = dialogflowcx_v3.QueryInput()
    query_input.text.text = 'text_value'
    query_input.language_code = 'language_code_value'
    request = dialogflowcx_v3.StreamingDetectIntentRequest(query_input=query_input)
    requests = [request]

    def request_generator():
        if False:
            for i in range(10):
                print('nop')
        for request in requests:
            yield request
    stream = await client.streaming_detect_intent(requests=request_generator())
    async for response in stream:
        print(response)