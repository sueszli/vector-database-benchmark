from google.cloud import retail_v2beta

def sample_rejoin_user_events():
    if False:
        i = 10
        return i + 15
    client = retail_v2beta.UserEventServiceClient()
    request = retail_v2beta.RejoinUserEventsRequest(parent='parent_value')
    operation = client.rejoin_user_events(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)