from google.cloud import discoveryengine_v1beta

def sample_recommend():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1beta.RecommendationServiceClient()
    user_event = discoveryengine_v1beta.UserEvent()
    user_event.event_type = 'event_type_value'
    user_event.user_pseudo_id = 'user_pseudo_id_value'
    request = discoveryengine_v1beta.RecommendRequest(serving_config='serving_config_value', user_event=user_event)
    response = client.recommend(request=request)
    print(response)