from google.cloud import retail_v2beta

def sample_predict():
    if False:
        while True:
            i = 10
    client = retail_v2beta.PredictionServiceClient()
    user_event = retail_v2beta.UserEvent()
    user_event.event_type = 'event_type_value'
    user_event.visitor_id = 'visitor_id_value'
    request = retail_v2beta.PredictRequest(placement='placement_value', user_event=user_event)
    response = client.predict(request=request)
    print(response)