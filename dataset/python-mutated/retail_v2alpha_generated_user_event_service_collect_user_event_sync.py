from google.cloud import retail_v2alpha

def sample_collect_user_event():
    if False:
        print('Hello World!')
    client = retail_v2alpha.UserEventServiceClient()
    request = retail_v2alpha.CollectUserEventRequest(prebuilt_rule='prebuilt_rule_value', parent='parent_value', user_event='user_event_value')
    response = client.collect_user_event(request=request)
    print(response)