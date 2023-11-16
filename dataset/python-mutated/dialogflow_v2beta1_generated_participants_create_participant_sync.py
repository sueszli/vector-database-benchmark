from google.cloud import dialogflow_v2beta1

def sample_create_participant():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.ParticipantsClient()
    request = dialogflow_v2beta1.CreateParticipantRequest(parent='parent_value')
    response = client.create_participant(request=request)
    print(response)