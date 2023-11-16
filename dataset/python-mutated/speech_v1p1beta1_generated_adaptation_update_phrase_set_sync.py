from google.cloud import speech_v1p1beta1

def sample_update_phrase_set():
    if False:
        return 10
    client = speech_v1p1beta1.AdaptationClient()
    request = speech_v1p1beta1.UpdatePhraseSetRequest()
    response = client.update_phrase_set(request=request)
    print(response)