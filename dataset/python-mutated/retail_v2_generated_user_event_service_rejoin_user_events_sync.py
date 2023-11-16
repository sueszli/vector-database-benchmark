from google.cloud import retail_v2

def sample_rejoin_user_events():
    if False:
        while True:
            i = 10
    client = retail_v2.UserEventServiceClient()
    request = retail_v2.RejoinUserEventsRequest(parent='parent_value')
    operation = client.rejoin_user_events(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)