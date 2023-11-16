from google.cloud import retail_v2alpha

def sample_purge_user_events():
    if False:
        i = 10
        return i + 15
    client = retail_v2alpha.UserEventServiceClient()
    request = retail_v2alpha.PurgeUserEventsRequest(parent='parent_value', filter='filter_value')
    operation = client.purge_user_events(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)