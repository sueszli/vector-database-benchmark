from google.cloud import speech_v2

async def sample_streaming_recognize():
    client = speech_v2.SpeechAsyncClient()
    request = speech_v2.StreamingRecognizeRequest(recognizer='recognizer_value')
    requests = [request]

    def request_generator():
        if False:
            return 10
        for request in requests:
            yield request
    stream = await client.streaming_recognize(requests=request_generator())
    async for response in stream:
        print(response)