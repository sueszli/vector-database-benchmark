from google.cloud import retail_v2beta

def sample_purge_user_events():
    if False:
        return 10
    client = retail_v2beta.UserEventServiceClient()
    request = retail_v2beta.PurgeUserEventsRequest(parent='parent_value', filter='filter_value')
    operation = client.purge_user_events(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)