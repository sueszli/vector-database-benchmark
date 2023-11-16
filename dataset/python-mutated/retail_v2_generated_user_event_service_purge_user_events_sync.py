from google.cloud import retail_v2

def sample_purge_user_events():
    if False:
        print('Hello World!')
    client = retail_v2.UserEventServiceClient()
    request = retail_v2.PurgeUserEventsRequest(parent='parent_value', filter='filter_value')
    operation = client.purge_user_events(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)