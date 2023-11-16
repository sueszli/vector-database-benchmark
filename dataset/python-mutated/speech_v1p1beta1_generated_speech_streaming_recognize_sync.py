from google.cloud import speech_v1p1beta1

def sample_streaming_recognize():
    if False:
        for i in range(10):
            print('nop')
    client = speech_v1p1beta1.SpeechClient()
    streaming_config = speech_v1p1beta1.StreamingRecognitionConfig()
    streaming_config.config.language_code = 'language_code_value'
    request = speech_v1p1beta1.StreamingRecognizeRequest(streaming_config=streaming_config)
    requests = [request]

    def request_generator():
        if False:
            while True:
                i = 10
        for request in requests:
            yield request
    stream = client.streaming_recognize(requests=request_generator())
    for response in stream:
        print(response)