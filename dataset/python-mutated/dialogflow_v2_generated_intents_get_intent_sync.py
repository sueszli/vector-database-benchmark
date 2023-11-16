from google.cloud import dialogflow_v2

def sample_get_intent():
    if False:
        return 10
    client = dialogflow_v2.IntentsClient()
    request = dialogflow_v2.GetIntentRequest(name='name_value')
    response = client.get_intent(request=request)
    print(response)