from google.cloud import dialogflow_v2beta1

def sample_get_fulfillment():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2beta1.FulfillmentsClient()
    request = dialogflow_v2beta1.GetFulfillmentRequest(name='name_value')
    response = client.get_fulfillment(request=request)
    print(response)