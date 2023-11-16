from google.cloud import dialogflowcx_v3beta1

def sample_get_intent():
    if False:
        return 10
    client = dialogflowcx_v3beta1.IntentsClient()
    request = dialogflowcx_v3beta1.GetIntentRequest(name='name_value')
    response = client.get_intent(request=request)
    print(response)