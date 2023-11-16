from google.cloud import retail_v2beta

def sample_collect_user_event():
    if False:
        return 10
    client = retail_v2beta.UserEventServiceClient()
    request = retail_v2beta.CollectUserEventRequest(prebuilt_rule='prebuilt_rule_value', parent='parent_value', user_event='user_event_value')
    response = client.collect_user_event(request=request)
    print(response)