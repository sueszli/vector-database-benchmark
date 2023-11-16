from google.cloud import speech_v1p1beta1

def sample_delete_phrase_set():
    if False:
        i = 10
        return i + 15
    client = speech_v1p1beta1.AdaptationClient()
    request = speech_v1p1beta1.DeletePhraseSetRequest(name='name_value')
    client.delete_phrase_set(request=request)