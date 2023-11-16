from google.cloud import videointelligence_v1p3beta1

async def sample_streaming_annotate_video():
    client = videointelligence_v1p3beta1.StreamingVideoIntelligenceServiceAsyncClient()
    request = videointelligence_v1p3beta1.StreamingAnnotateVideoRequest()
    requests = [request]

    def request_generator():
        if False:
            for i in range(10):
                print('nop')
        for request in requests:
            yield request
    stream = await client.streaming_annotate_video(requests=request_generator())
    async for response in stream:
        print(response)