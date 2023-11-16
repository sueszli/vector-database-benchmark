from google.cloud import speech_v2

def sample_streaming_recognize():
    if False:
        for i in range(10):
            print('nop')
    client = speech_v2.SpeechClient()
    request = speech_v2.StreamingRecognizeRequest(recognizer='recognizer_value')
    requests = [request]

    def request_generator():
        if False:
            return 10
        for request in requests:
            yield request
    stream = client.streaming_recognize(requests=request_generator())
    for response in stream:
        print(response)