from google.cloud import discoveryengine_v1

def sample_collect_user_event():
    if False:
        return 10
    client = discoveryengine_v1.UserEventServiceClient()
    request = discoveryengine_v1.CollectUserEventRequest(parent='parent_value', user_event='user_event_value')
    response = client.collect_user_event(request=request)
    print(response)