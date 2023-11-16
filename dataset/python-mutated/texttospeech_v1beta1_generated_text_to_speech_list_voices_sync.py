from google.cloud import texttospeech_v1beta1

def sample_list_voices():
    if False:
        while True:
            i = 10
    client = texttospeech_v1beta1.TextToSpeechClient()
    request = texttospeech_v1beta1.ListVoicesRequest()
    response = client.list_voices(request=request)
    print(response)