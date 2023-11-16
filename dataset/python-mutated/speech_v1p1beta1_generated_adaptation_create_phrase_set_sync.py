from google.cloud import speech_v1p1beta1

def sample_create_phrase_set():
    if False:
        while True:
            i = 10
    client = speech_v1p1beta1.AdaptationClient()
    request = speech_v1p1beta1.CreatePhraseSetRequest(parent='parent_value', phrase_set_id='phrase_set_id_value')
    response = client.create_phrase_set(request=request)
    print(response)