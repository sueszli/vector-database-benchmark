from google.cloud import speech_v1

def sample_delete_phrase_set():
    if False:
        print('Hello World!')
    client = speech_v1.AdaptationClient()
    request = speech_v1.DeletePhraseSetRequest(name='name_value')
    client.delete_phrase_set(request=request)