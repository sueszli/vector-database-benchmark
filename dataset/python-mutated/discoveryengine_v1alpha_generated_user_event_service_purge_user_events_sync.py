from google.cloud import discoveryengine_v1alpha

def sample_purge_user_events():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1alpha.UserEventServiceClient()
    request = discoveryengine_v1alpha.PurgeUserEventsRequest(parent='parent_value', filter='filter_value')
    operation = client.purge_user_events(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)