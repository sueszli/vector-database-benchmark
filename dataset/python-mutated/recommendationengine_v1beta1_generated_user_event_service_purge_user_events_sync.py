from google.cloud import recommendationengine_v1beta1

def sample_purge_user_events():
    if False:
        return 10
    client = recommendationengine_v1beta1.UserEventServiceClient()
    request = recommendationengine_v1beta1.PurgeUserEventsRequest(parent='parent_value', filter='filter_value')
    operation = client.purge_user_events(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)