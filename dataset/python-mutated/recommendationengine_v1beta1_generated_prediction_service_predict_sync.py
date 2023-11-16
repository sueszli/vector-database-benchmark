from google.cloud import recommendationengine_v1beta1

def sample_predict():
    if False:
        print('Hello World!')
    client = recommendationengine_v1beta1.PredictionServiceClient()
    user_event = recommendationengine_v1beta1.UserEvent()
    user_event.event_type = 'event_type_value'
    user_event.user_info.visitor_id = 'visitor_id_value'
    request = recommendationengine_v1beta1.PredictRequest(name='name_value', user_event=user_event)
    page_result = client.predict(request=request)
    for response in page_result:
        print(response)