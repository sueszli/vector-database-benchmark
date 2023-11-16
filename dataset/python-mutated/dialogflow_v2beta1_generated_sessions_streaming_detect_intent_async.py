from google.cloud import dialogflow_v2beta1

async def sample_streaming_detect_intent():
    client = dialogflow_v2beta1.SessionsAsyncClient()
    query_input = dialogflow_v2beta1.QueryInput()
    query_input.audio_config.audio_encoding = 'AUDIO_ENCODING_SPEEX_WITH_HEADER_BYTE'
    query_input.audio_config.sample_rate_hertz = 1817
    query_input.audio_config.language_code = 'language_code_value'
    request = dialogflow_v2beta1.StreamingDetectIntentRequest(session='session_value', query_input=query_input)
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