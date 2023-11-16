from google.cloud import speech_v1

def sample_list_phrase_set():
    if False:
        for i in range(10):
            print('nop')
    client = speech_v1.AdaptationClient()
    request = speech_v1.ListPhraseSetRequest(parent='parent_value')
    page_result = client.list_phrase_set(request=request)
    for response in page_result:
        print(response)