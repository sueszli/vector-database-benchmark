from google.cloud import speech_v1p1beta1

async def sample_streaming_recognize():
    client = speech_v1p1beta1.SpeechAsyncClient()
    streaming_config = speech_v1p1beta1.StreamingRecognitionConfig()
    streaming_config.config.language_code = 'language_code_value'
    request = speech_v1p1beta1.StreamingRecognizeRequest(streaming_config=streaming_config)
    requests = [request]

    def request_generator():
        if False:
            return 10
        for request in requests:
            yield request
    stream = await client.streaming_recognize(requests=request_generator())
    async for response in stream:
        print(response)