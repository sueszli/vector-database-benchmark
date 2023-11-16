from google.cloud import speech_v1

def sample_update_phrase_set():
    if False:
        print('Hello World!')
    client = speech_v1.AdaptationClient()
    request = speech_v1.UpdatePhraseSetRequest()
    response = client.update_phrase_set(request=request)
    print(response)