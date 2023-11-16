from google.cloud import speech_v2

def sample_list_recognizers():
    if False:
        print('Hello World!')
    client = speech_v2.SpeechClient()
    request = speech_v2.ListRecognizersRequest(parent='parent_value')
    page_result = client.list_recognizers(request=request)
    for response in page_result:
        print(response)