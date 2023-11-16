from google.cloud import speech_v1

def sample_get_phrase_set():
    if False:
        for i in range(10):
            print('nop')
    client = speech_v1.AdaptationClient()
    request = speech_v1.GetPhraseSetRequest(name='name_value')
    response = client.get_phrase_set(request=request)
    print(response)