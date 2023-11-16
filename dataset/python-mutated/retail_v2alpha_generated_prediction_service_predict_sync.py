from google.cloud import retail_v2alpha

def sample_predict():
    if False:
        return 10
    client = retail_v2alpha.PredictionServiceClient()
    user_event = retail_v2alpha.UserEvent()
    user_event.event_type = 'event_type_value'
    user_event.visitor_id = 'visitor_id_value'
    request = retail_v2alpha.PredictRequest(placement='placement_value', user_event=user_event)
    response = client.predict(request=request)
    print(response)