from google.cloud import dialogflow_v2beta1

def sample_get_intent():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.IntentsClient()
    request = dialogflow_v2beta1.GetIntentRequest(name='name_value')
    response = client.get_intent(request=request)
    print(response)