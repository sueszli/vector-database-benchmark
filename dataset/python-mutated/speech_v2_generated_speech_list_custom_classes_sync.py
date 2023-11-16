from google.cloud import speech_v2

def sample_list_custom_classes():
    if False:
        print('Hello World!')
    client = speech_v2.SpeechClient()
    request = speech_v2.ListCustomClassesRequest(parent='parent_value')
    page_result = client.list_custom_classes(request=request)
    for response in page_result:
        print(response)