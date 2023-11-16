from google.cloud import dialogflow_v2

def sample_get_participant():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.ParticipantsClient()
    request = dialogflow_v2.GetParticipantRequest(name='name_value')
    response = client.get_participant(request=request)
    print(response)