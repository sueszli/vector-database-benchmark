from google.cloud import dialogflow_v2beta1

def sample_update_participant():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.ParticipantsClient()
    request = dialogflow_v2beta1.UpdateParticipantRequest()
    response = client.update_participant(request=request)
    print(response)