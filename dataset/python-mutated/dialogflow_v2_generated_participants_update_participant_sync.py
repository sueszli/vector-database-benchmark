from google.cloud import dialogflow_v2

def sample_update_participant():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.ParticipantsClient()
    request = dialogflow_v2.UpdateParticipantRequest()
    response = client.update_participant(request=request)
    print(response)