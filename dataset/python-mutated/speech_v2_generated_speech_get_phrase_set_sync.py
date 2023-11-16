from google.cloud import speech_v2

def sample_get_phrase_set():
    if False:
        i = 10
        return i + 15
    client = speech_v2.SpeechClient()
    request = speech_v2.GetPhraseSetRequest(name='name_value')
    response = client.get_phrase_set(request=request)
    print(response)