from google.cloud import speech_v1p1beta1

def sample_list_phrase_set():
    if False:
        print('Hello World!')
    client = speech_v1p1beta1.AdaptationClient()
    request = speech_v1p1beta1.ListPhraseSetRequest(parent='parent_value')
    page_result = client.list_phrase_set(request=request)
    for response in page_result:
        print(response)