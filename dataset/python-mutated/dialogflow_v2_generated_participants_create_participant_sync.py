from google.cloud import dialogflow_v2

def sample_create_participant():
    if False:
        print('Hello World!')
    client = dialogflow_v2.ParticipantsClient()
    request = dialogflow_v2.CreateParticipantRequest(parent='parent_value')
    response = client.create_participant(request=request)
    print(response)