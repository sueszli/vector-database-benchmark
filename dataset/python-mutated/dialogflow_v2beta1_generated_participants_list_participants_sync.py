from google.cloud import dialogflow_v2beta1

def sample_list_participants():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.ParticipantsClient()
    request = dialogflow_v2beta1.ListParticipantsRequest(parent='parent_value')
    page_result = client.list_participants(request=request)
    for response in page_result:
        print(response)