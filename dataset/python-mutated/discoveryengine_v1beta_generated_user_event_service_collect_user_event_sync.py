from google.cloud import discoveryengine_v1beta

def sample_collect_user_event():
    if False:
        print('Hello World!')
    client = discoveryengine_v1beta.UserEventServiceClient()
    request = discoveryengine_v1beta.CollectUserEventRequest(parent='parent_value', user_event='user_event_value')
    response = client.collect_user_event(request=request)
    print(response)