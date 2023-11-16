from google.cloud import recommendationengine_v1beta1

def sample_list_user_events():
    if False:
        return 10
    client = recommendationengine_v1beta1.UserEventServiceClient()
    request = recommendationengine_v1beta1.ListUserEventsRequest(parent='parent_value')
    page_result = client.list_user_events(request=request)
    for response in page_result:
        print(response)