from google.cloud import texttospeech_v1

def sample_list_voices():
    if False:
        for i in range(10):
            print('nop')
    client = texttospeech_v1.TextToSpeechClient()
    request = texttospeech_v1.ListVoicesRequest()
    response = client.list_voices(request=request)
    print(response)