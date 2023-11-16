from google.cloud import speech_v2

def sample_list_phrase_sets():
    if False:
        return 10
    client = speech_v2.SpeechClient()
    request = speech_v2.ListPhraseSetsRequest(parent='parent_value')
    page_result = client.list_phrase_sets(request=request)
    for response in page_result:
        print(response)