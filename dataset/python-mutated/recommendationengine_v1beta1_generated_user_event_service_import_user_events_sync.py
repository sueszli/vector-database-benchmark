from google.cloud import recommendationengine_v1beta1

def sample_import_user_events():
    if False:
        print('Hello World!')
    client = recommendationengine_v1beta1.UserEventServiceClient()
    request = recommendationengine_v1beta1.ImportUserEventsRequest(parent='parent_value')
    operation = client.import_user_events(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)