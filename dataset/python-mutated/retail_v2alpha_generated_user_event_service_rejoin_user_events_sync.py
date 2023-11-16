from google.cloud import retail_v2alpha

def sample_rejoin_user_events():
    if False:
        while True:
            i = 10
    client = retail_v2alpha.UserEventServiceClient()
    request = retail_v2alpha.RejoinUserEventsRequest(parent='parent_value')
    operation = client.rejoin_user_events(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)