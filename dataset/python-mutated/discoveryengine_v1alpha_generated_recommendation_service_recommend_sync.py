from google.cloud import discoveryengine_v1alpha

def sample_recommend():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1alpha.RecommendationServiceClient()
    user_event = discoveryengine_v1alpha.UserEvent()
    user_event.event_type = 'event_type_value'
    user_event.user_pseudo_id = 'user_pseudo_id_value'
    request = discoveryengine_v1alpha.RecommendRequest(serving_config='serving_config_value', user_event=user_event)
    response = client.recommend(request=request)
    print(response)