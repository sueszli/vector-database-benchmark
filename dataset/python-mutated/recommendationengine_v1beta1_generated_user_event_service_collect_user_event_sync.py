from google.cloud import recommendationengine_v1beta1

def sample_collect_user_event():
    if False:
        i = 10
        return i + 15
    client = recommendationengine_v1beta1.UserEventServiceClient()
    request = recommendationengine_v1beta1.CollectUserEventRequest(parent='parent_value', user_event='user_event_value')
    response = client.collect_user_event(request=request)
    print(response)