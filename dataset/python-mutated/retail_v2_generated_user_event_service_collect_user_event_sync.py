from google.cloud import retail_v2

def sample_collect_user_event():
    if False:
        while True:
            i = 10
    client = retail_v2.UserEventServiceClient()
    request = retail_v2.CollectUserEventRequest(prebuilt_rule='prebuilt_rule_value', parent='parent_value', user_event='user_event_value')
    response = client.collect_user_event(request=request)
    print(response)