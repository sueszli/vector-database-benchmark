from google.cloud import videointelligence_v1p3beta1

def sample_streaming_annotate_video():
    if False:
        print('Hello World!')
    client = videointelligence_v1p3beta1.StreamingVideoIntelligenceServiceClient()
    request = videointelligence_v1p3beta1.StreamingAnnotateVideoRequest()
    requests = [request]

    def request_generator():
        if False:
            while True:
                i = 10
        for request in requests:
            yield request
    stream = client.streaming_annotate_video(requests=request_generator())
    for response in stream:
        print(response)