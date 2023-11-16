from google.cloud import recommendationengine_v1beta1

def sample_write_user_event():
    if False:
        for i in range(10):
            print('nop')
    client = recommendationengine_v1beta1.UserEventServiceClient()
    user_event = recommendationengine_v1beta1.UserEvent()
    user_event.event_type = 'event_type_value'
    user_event.user_info.visitor_id = 'visitor_id_value'
    request = recommendationengine_v1beta1.WriteUserEventRequest(parent='parent_value', user_event=user_event)
    response = client.write_user_event(request=request)
    print(response)