from google.cloud import dialogflow_v2beta1

def sample_get_participant():
    if False:
        return 10
    client = dialogflow_v2beta1.ParticipantsClient()
    request = dialogflow_v2beta1.GetParticipantRequest(name='name_value')
    response = client.get_participant(request=request)
    print(response)