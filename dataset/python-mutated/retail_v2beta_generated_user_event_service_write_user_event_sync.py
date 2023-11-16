from google.cloud import retail_v2beta

def sample_write_user_event():
    if False:
        i = 10
        return i + 15
    client = retail_v2beta.UserEventServiceClient()
    user_event = retail_v2beta.UserEvent()
    user_event.event_type = 'event_type_value'
    user_event.visitor_id = 'visitor_id_value'
    request = retail_v2beta.WriteUserEventRequest(parent='parent_value', user_event=user_event)
    response = client.write_user_event(request=request)
    print(response)